// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// (hecke-extensions slot 22, BIN-3) the degree-1 amalgam reader -- see AmalgamReader.h.
#include "AmalgamReader.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <type_traits>
#include <utility>

#include "../../../ModArithmetic/Binary/Ring2k.h"
#include "../../../ModArithmetic/Binary/TritMatrix.h"
#include "../../../ModArithmetic/DenseElim.h"
#include "../../../ModArithmetic/EngineCheck.h"
#include "../../../ModArithmetic/EnvLever.h"
#include "../../../ModArithmetic/LadicInduction.h"
#include "../../../ModArithmetic/ModCtx.h"

namespace l23 {

// ---------------------------------------------------------------------------
// Divisors
// ---------------------------------------------------------------------------
uint32_t Divisors::units() const {
    return static_cast<uint32_t>(std::count(finite.begin(), finite.end(), 0u));
}
std::string Divisors::ladic2() const {
    std::string s;
    for (uint32_t x : finite) {
        if (x == 0) continue;
        if (!s.empty()) s += ",";
        s += std::to_string(x);
    }
    if (s.empty()) s = "-";
    if (at_least) s += "+[>=" + std::to_string(cap) + "]^" + std::to_string(at_least);
    return s;
}
std::string Divisors::census() const {
    std::string s;
    for (uint32_t x : finite) s += (s.empty() ? "" : ",") + std::to_string(x);
    for (uint32_t i = 0; i < at_least; ++i) s += (s.empty() ? ">=" : ",>=") + std::to_string(cap);
    return s.empty() ? "-" : s;
}

std::string LatticeCell::ladic2_line() const {
    std::ostringstream o;
    o << "ell=" << ell << " n=" << n << " rankL1=" << rank_lambda << " T1=[";
    for (size_t i = 0; i < torsion.size(); ++i) o << (i ? ", " : "") << torsion[i];
    o << "]";
    for (const auto& kv : ed_tp) o << " p" << kv.first << ":" << kv.second.ladic2();
    for (const auto& kv : ed_eis) o << " Eis" << kv.first << ":" << kv.second.ladic2();
    o << " Tl:" << ed_tell.ladic2() << " Tl2-Tl:" << ed_tell2.ladic2();
    return o.str();
}

std::string OddnRow::line() const {
    std::string s = std::to_string(n);
    for (uint32_t x : v) s += " " + std::to_string(x);
    return s;
}

uint32_t dim_modular_forms(uint32_t k) {
    if (k % 2 || k == 2) return 0;
    return k / 12 + (k % 12 == 2 ? 0 : 1);
}
uint32_t expected_rank_lambda(uint32_t n) {
    if (n % 2) return 0;
    return 2 * dim_modular_forms(n + 2) - 1;
}

namespace {

namespace r2 = engine::ring2k;
namespace ladic = engine::ladic;
using r2::u128;

// ---------------------------------------------------------------------------
// 2 x 2 integer matrices and the Euclid word (mp4_oracle.py's gmul / ginv / word / cosets / coset_index)
// ---------------------------------------------------------------------------
struct G2 { int64_t a, b, c, d; };
G2 gmul(const G2& g, const G2& h) {
    return {g.a * h.a + g.b * h.c, g.a * h.b + g.b * h.d, g.c * h.a + g.d * h.c, g.c * h.b + g.d * h.d};
}
G2 ginv(const G2& g) {
    ENGINE_CHECK_MSG(g.a * g.d - g.b * g.c == 1, "l23::ginv: not in SL_2(Z)");
    return {g.d, -g.b, -g.c, g.a};
}
bool geq(const G2& g, const G2& h) { return g.a == h.a && g.b == h.b && g.c == h.c && g.d == h.d; }
const G2 kS{0, -1, 1, 0}, kT{1, 1, 0, 1}, kI{1, 0, 0, 1}, kNegI{-1, 0, 0, -1};
const G2 kSinv{0, 1, -1, 0};
const G2 kTinv{1, -1, 0, 1};
const G2 kU{0, -1, 1, 1};   // S T
int64_t fdiv(int64_t a, int64_t b) {   // Python's a // b
    int64_t q = a / b;
    if ((a % b != 0) && ((a < 0) != (b < 0))) --q;
    return q;
}
int64_t pmod(int64_t a, int64_t p) { a %= p; return a < 0 ? a + p : a; }

enum class Letter : uint8_t { S, T, Ti };

std::vector<Letter> letters(const G2& g) {
    std::vector<std::pair<char, int64_t>> out;
    G2 cur = g;
    while (cur.c != 0) {
        const int64_t q = fdiv(cur.a, cur.c);
        if (q) out.push_back({'T', q});
        out.push_back({'S', 1});
        const G2 nxt = gmul(kSinv, gmul(G2{1, -q, 0, 1}, cur));
        ENGINE_CHECK_MSG((nxt.c < 0 ? -nxt.c : nxt.c) < (cur.c < 0 ? -cur.c : cur.c), "l23::letters: Euclid step");
        cur = nxt;
    }
    ENGINE_CHECK_MSG(cur.c == 0 && cur.a * cur.d == 1, "l23::letters: not in SL_2(Z)");
    if (cur.a == -1) {
        out.push_back({'S', 2});
        cur = gmul(kNegI, cur);
    }
    if (cur.b) out.push_back({'T', cur.b});
    std::vector<Letter> L;
    G2 prod = kI;
    for (const auto& sk : out) {
        if (sk.first == 'S') {
            for (int64_t i = 0; i < pmod(sk.second, 4); ++i) { L.push_back(Letter::S); prod = gmul(prod, kS); }
        } else {
            const int64_t k = sk.second;
            for (int64_t i = 0; i < (k < 0 ? -k : k); ++i) {
                L.push_back(k > 0 ? Letter::T : Letter::Ti);
                prod = gmul(prod, k > 0 ? kT : kTinv);
            }
        }
    }
    ENGINE_CHECK_MSG(geq(prod, g), "l23::letters: the word does not multiply to g");
    return L;
}

uint32_t coset_index(const G2& gm, uint32_t p) {
    const int64_t c = pmod(gm.c, p), d = pmod(gm.d, p);
    if (d == 0) return p;
    int64_t inv = 1, b = d, e = p - 2;
    while (e) { if (e & 1) inv = inv * b % p; b = b * b % p; e >>= 1; }
    return static_cast<uint32_t>(c * inv % p);
}
std::vector<G2> coset_reps(uint32_t p) {
    std::vector<G2> r;
    for (uint32_t k = 0; k < p; ++k) r.push_back(G2{1, 0, static_cast<int64_t>(k), 1});
    r.push_back(kS);
    return r;
}

// ---------------------------------------------------------------------------
// Dense row-major matrices over a ring element type
// ---------------------------------------------------------------------------
template<class E>
struct RMat {
    uint32_t rows = 0, cols = 0;
    std::vector<E> a;
    RMat() = default;
    RMat(uint32_t r, uint32_t c, const E& z) : rows(r), cols(c), a(static_cast<size_t>(r) * c, z) {}
    E* row(uint32_t i) { return a.data() + static_cast<size_t>(i) * cols; }
    const E* row(uint32_t i) const { return a.data() + static_cast<size_t>(i) * cols; }
    E& at(uint32_t i, uint32_t j) { return a[static_cast<size_t>(i) * cols + j]; }
    const E& at(uint32_t i, uint32_t j) const { return a[static_cast<size_t>(i) * cols + j]; }
};

template<class E>
struct HowellOut {
    RMat<E> H;   // h x n
    std::vector<uint32_t> pivots, pivot_v;
};

// ---------------------------------------------------------------------------
// The rings.  Ops2<E>: Z/2^k on a ring2k rung (arithmetic mod 2^W by wraparound, read mod 2^k).
// ---------------------------------------------------------------------------
template<class E>
struct Ops2 {
    using Tr = r2::Traits<E>;
    using Elt = E;
    static constexpr uint32_t kEll = 2;
    uint32_t K;
    explicit Ops2(uint32_t K_) : K(K_) { ENGINE_CHECK_MSG(K >= 1 && K <= Tr::kBits, "l23::Ops2: K exceeds the rung"); }
    std::string label() const { return r2::rung_label(r2::select_rung(K < 2 ? 2 : K)); }
    Elt zero() const { return Tr::zero(); }
    Elt one() const { return Tr::one(); }
    Elt from_i64(int64_t v) const {
        const uint64_t u = v < 0 ? uint64_t(0) - static_cast<uint64_t>(v) : static_cast<uint64_t>(v);
        const Elt x = Tr::from_u64(u);
        return v < 0 ? Tr::neg(x) : x;
    }
    Elt add(const Elt& x, const Elt& y) const { return Tr::add(x, y); }
    Elt sub(const Elt& x, const Elt& y) const { return Tr::sub(x, y); }
    Elt mul(const Elt& x, const Elt& y) const { return Tr::mul(x, y); }
    Elt neg(const Elt& x) const { return Tr::neg(x); }
    uint32_t val(const Elt& x, uint32_t k) const { return Tr::val(x, k); }
    bool zero_mod(const Elt& x, uint32_t k) const { return Tr::zero_mod(x, k); }
    Elt reduce(const Elt& x, uint32_t k) const { return Tr::mask(x, k); }
    Elt inv2() const { ENGINE_CHECK_MSG(false, "l23::Ops2: 2 is not a unit"); return zero(); }
    Elt inv_unit(const Elt& u) const { ENGINE_CHECK_MSG(Tr::odd(u), "l23::Ops2::inv_unit: not a unit"); return Tr::inv_odd(u); }
    // x mod 2^k as an integer (k <= 64).
    uint64_t lowval(const Elt& x, uint32_t k) const {
        ENGINE_CHECK_MSG(k <= 64, "l23::Ops2::lowval: k <= 64");
        uint64_t w[1] = {0};
        Tr::to_words(Tr::mask(x, k), w, 1);
        return w[0];
    }

    static r2::Mat<E> to_r2(RMat<E> A) {
        r2::Mat<E> M;
        M.rows = A.rows;
        M.cols = A.cols;
        M.a = std::move(A.a);
        return M;
    }
    static RMat<E> from_r2(r2::Mat<E> M) {
        RMat<E> A;
        A.rows = M.rows;
        A.cols = M.cols;
        A.a = std::move(M.a);
        return A;
    }
    HowellOut<E> howell(RMat<E> A, uint32_t k) const {
        r2::HowellForm<E> h = r2::howell<Tr>(to_r2(std::move(A)), k);
        return {from_r2(std::move(h.H)), h.pivots, h.pivot_v};
    }
    HowellOut<E> kernel_module(const RMat<E>& A, uint32_t k) const {
        r2::HowellForm<E> h = r2::kernel_module<Tr>(to_r2(A), k);
        return {from_r2(std::move(h.H)), h.pivots, h.pivot_v};
    }
    std::vector<uint32_t> smith(RMat<E> A, uint32_t k) const {
        if (A.rows == 0 || A.cols == 0) return {};
        return r2::smith<Tr>(to_r2(std::move(A)), k).sigma;
    }
    // sigma(A) at K through ring2k::smith_adaptive with the rank over Q as the hint (one run: k_min = k_cap = K).
    std::vector<uint32_t> smith_hinted(const RMat<E>& A, int64_t rank_hint, std::vector<std::string>& log, bool& verified) const {
        ENGINE_CHECK_MSG(K >= 2, "l23::Ops2::smith_hinted: K >= 2");
        auto produce = [&](uint32_t Kp) { return r2::to_wide(to_r2(A), Kp); };
        const r2::AdaptiveSmith s = r2::smith_adaptive(produce, K, 0, K, rank_hint);
        for (const std::string& l : s.log) log.push_back(l);
        verified = s.verified;
        return s.smith.sigma;
    }
};

// Ops3<L>: Z/3^k on the engine's digit planes (engine::ladic::Ring, base 3^39, L digits), k >= 2 through
// engine::ladic (ModCtx::prime_power for one digit), k = 1 through engine::elim over F_3.
template<uint32_t L>
struct D3 { uint64_t w[L < 2 ? 2 : L]; };   // (one spare word at L = 1 keeps -Warray-bounds quiet; never read)

template<uint32_t L>
struct Ops3 {
    using W = uint64_t;
    using Elt = D3<L>;
    static constexpr uint32_t kEll = 3;
    uint32_t K;
    ladic::DigitLayout<W> lay;
    std::shared_ptr<ladic::Ring<W>> ring;
    explicit Ops3(uint32_t K_) : K(K_), lay(ladic::DigitLayout<W>::make(3, K_ < 2 ? 2 : K_)) {
        ENGINE_CHECK_MSG(lay.L == L, "l23::Ops3: digit count mismatch");
        ring = std::make_shared<ladic::Ring<W>>(lay);
    }
    std::string label() const { return "Z/3^K(L=" + std::to_string(L) + ")"; }
    Elt zero() const { Elt z{}; return z; }
    Elt one() const { Elt z = zero(); z.w[0] = 1; return z; }
    Elt from_i64(int64_t v) const {
        Elt z{};
        const uint64_t u = v < 0 ? uint64_t(0) - static_cast<uint64_t>(v) : static_cast<uint64_t>(v);
        ring->from_u64(z.w, u);
        if (v < 0) ring->neg(z.w, z.w);
        return z;
    }
    Elt add(const Elt& x, const Elt& y) const { Elt z{}; ring->add(z.w, x.w, y.w); return z; }
    Elt sub(const Elt& x, const Elt& y) const { Elt z{}; ring->sub(z.w, x.w, y.w); return z; }
    Elt mul(const Elt& x, const Elt& y) const { Elt z{}; ring->mul(z.w, x.w, y.w); return z; }
    Elt neg(const Elt& x) const { Elt z{}; ring->neg(z.w, x.w); return z; }
    uint32_t val(const Elt& x, uint32_t k) const { const uint32_t v = ring->valuation(x.w); return v < k ? v : k; }
    bool zero_mod(const Elt& x, uint32_t k) const { return ring->valuation(x.w) >= k; }
    Elt reduce(const Elt& x, uint32_t k) const { Elt z{}; ring->low(z.w, x.w, k); return z; }
    Elt inv2() const { Elt two = from_i64(2), z{}; ring->inv(z.w, two.w); return z; }
    Elt inv_unit(const Elt& u) const { Elt z{}; ring->inv(z.w, u.w); return z; }
    // x mod 3^k as an integer (k <= 39, one digit).
    uint64_t lowval(const Elt& x, uint32_t k) const {
        ENGINE_CHECK_MSG(k >= 1 && k <= 39, "l23::Ops3::lowval: 1 <= k <= 39");
        const uint64_t v = reduce(x, k).w[0];
        uint64_t m = 1;
        for (uint32_t i = 0; i < k; ++i) m *= 3;
        ENGINE_CHECK_MSG(v < m, "l23::Ops3::lowval: reduce is not canonical");
        return v;
    }

    ladic::LadicMat<W> to_ladic(const RMat<Elt>& A, uint32_t k) const {
        const ladic::DigitLayout<W> lk = ladic::DigitLayout<W>::make(3, k);
        ladic::LadicMat<W> X(lk, A.rows, A.cols);
        for (uint32_t i = 0; i < A.rows; ++i)
            for (uint32_t j = 0; j < A.cols; ++j) {
                const Elt r = reduce(A.at(i, j), k);
                X.set(i, j, r.w);   // copies lk.L <= L digits (the top one already reduced)
            }
        return X;
    }
    RMat<Elt> from_ladic(const ladic::LadicMat<W>& X) const {
        RMat<Elt> A(X.rows(), X.cols(), zero());
        std::vector<W> t(X.layout().L);
        for (uint32_t i = 0; i < X.rows(); ++i)
            for (uint32_t j = 0; j < X.cols(); ++j) {
                X.get(i, j, t.data());
                Elt& e = A.at(i, j);
                for (uint32_t q = 0; q < L; ++q) e.w[q] = q < t.size() ? t[q] : 0;
            }
        return A;
    }
    // F_3 (k = 1) through engine::elim.
    engine::DenseBuf<uint64_t> to_f3(const RMat<Elt>& A) const {
        engine::DenseBuf<uint64_t> b(A.rows, A.cols);
        for (uint32_t i = 0; i < A.rows; ++i)
            for (uint32_t j = 0; j < A.cols; ++j) b.at(i, j) = reduce(A.at(i, j), 1).w[0];
        return b;
    }
    HowellOut<Elt> rref_rows(engine::DenseBuf<uint64_t>& b) const {
        const engine::ModCtx<uint64_t> F(3);
        HowellOut<Elt> h;
        if (b.rows == 0 || b.cols == 0) { h.H = RMat<Elt>(0, b.cols, zero()); return h; }
        const engine::elim::RrefResult rr = engine::elim::rref(F, engine::DenseView<uint64_t>(b));
        h.H = RMat<Elt>(rr.rank, b.cols, zero());
        for (uint32_t i = 0; i < rr.rank; ++i)
            for (uint32_t j = 0; j < b.cols; ++j) h.H.at(i, j).w[0] = b.at(i, j);
        h.pivots = rr.pivots;
        h.pivot_v.assign(rr.rank, 0);
        return h;
    }
    HowellOut<Elt> howell(RMat<Elt> A, uint32_t k) const {
        if (A.rows == 0 || A.cols == 0) { HowellOut<Elt> h; h.H = RMat<Elt>(0, A.cols, zero()); return h; }
        if (k == 1) { engine::DenseBuf<uint64_t> b = to_f3(A); return rref_rows(b); }
        const ladic::Howell<W> h = ladic::howell(to_ladic(A, k));
        return {from_ladic(h.H), h.pivots, h.pivot_v};
    }
    HowellOut<Elt> kernel_module(const RMat<Elt>& A, uint32_t k) const {
        if (k == 1) {
            const engine::ModCtx<uint64_t> F(3);
            engine::DenseBuf<uint64_t> b = to_f3(A);
            const engine::elim::RrefResult rr = engine::elim::rref(F, engine::DenseView<uint64_t>(b));
            engine::DenseBuf<uint64_t> kb = engine::elim::kernel_basis(F, engine::cview(b), rr.pivots);
            return rref_rows(kb);
        }
        const ladic::Howell<W> h = ladic::kernel_module(to_ladic(A, k));
        return {from_ladic(h.H), h.pivots, h.pivot_v};
    }
    std::vector<uint32_t> smith(RMat<Elt> A, uint32_t k) const {
        if (A.rows == 0 || A.cols == 0) return {};
        if (k == 1) {
            const engine::ModCtx<uint64_t> F(3);
            engine::DenseBuf<uint64_t> b = to_f3(A);
            const engine::elim::RrefResult rr = engine::elim::rref(F, engine::DenseView<uint64_t>(b));
            return std::vector<uint32_t>(rr.rank, 0u);
        }
        const std::vector<uint32_t> f = ladic::invariant_factors(to_ladic(A, k));   // k - sigma, descending
        std::vector<uint32_t> s;
        for (uint32_t x : f) s.push_back(k - x);
        std::sort(s.begin(), s.end());
        return s;
    }
    std::vector<uint32_t> smith_hinted(const RMat<Elt>& A, int64_t rank_hint, std::vector<std::string>& log, bool& verified) const {
        std::vector<uint32_t> s = smith(A, K);
        const uint32_t mn = A.rows < A.cols ? A.rows : A.cols;
        verified = rank_hint >= 0 && static_cast<int64_t>(s.size()) == rank_hint;
        char buf[256];
        std::snprintf(buf, sizeof(buf), "BIN3_SMITH_A ell=3 K=%u invariants=%zu zeros=%u sigma_max=%u rank_hint=%lld -> %s", K,
                      s.size(), mn - static_cast<uint32_t>(s.size()), s.empty() ? 0u : s.back(),
                      static_cast<long long>(rank_hint), verified ? "verified" : "rank hint NOT met (an invariant >= K)");
        log.emplace_back(buf);
        return s;
    }
};

// (slot-23 TER-1) Ops3F: F_3 for k = 1 at ell = 3 on uint8_t entries in {0, 1, 2} (see AmalgamReader.h).  It carries
// the row kernels (kF3Rows: Rho uses the byte kernels and the characteristic-3 Taylor shift) and eliminates on the
// bit-sliced binary::TritMatrix; its howell / kernel_module / smith return exactly what Ops3<L> returns at k = 1 (the
// RREF rows and pivots, the RREF of the D5b kernel basis, the rank): unique objects, so no value can change.
struct Ops3F {
    using Elt = uint8_t;
    static constexpr uint32_t kEll = 3;
    static constexpr bool kF3Rows = true;
    uint32_t K;
    explicit Ops3F(uint32_t K_) : K(K_) { ENGINE_CHECK_MSG(K_ == 1, "l23::Ops3F: F_3 only (K = 1)"); }
    std::string label() const { return "F_3(TritMatrix)"; }
    Elt zero() const { return 0; }
    Elt one() const { return 1; }
    Elt from_i64(int64_t v) const { return static_cast<Elt>(pmod(v, 3)); }
    Elt add(Elt x, Elt y) const { const unsigned s = unsigned(x) + y; return static_cast<Elt>(s >= 3 ? s - 3 : s); }
    Elt sub(Elt x, Elt y) const { const unsigned s = unsigned(x) + 3u - y; return static_cast<Elt>(s >= 3 ? s - 3 : s); }
    Elt mul(Elt x, Elt y) const { return static_cast<Elt>((unsigned(x) * y) % 3u); }
    Elt neg(Elt x) const { return static_cast<Elt>(x ? 3 - x : 0); }
    uint32_t val(Elt x, uint32_t k) const { return x ? 0 : k; }
    bool zero_mod(Elt x, uint32_t k) const { return k == 0 || x == 0; }
    Elt reduce(Elt x, uint32_t) const { return x; }
    Elt inv2() const { return 2; }
    Elt inv_unit(Elt u) const { ENGINE_CHECK_MSG(u == 1 || u == 2, "l23::Ops3F::inv_unit: not a unit"); return u; }
    uint64_t lowval(Elt x, uint32_t k) const { ENGINE_CHECK_MSG(k == 1, "l23::Ops3F::lowval: k = 1"); return x; }
    // The row kernels (Rho / mat_add / mat_sub).
    void row_add(Elt* dst, const Elt* src, size_t r) const { engine::binary::f3_add_u8(dst, src, r); }
    void row_sub(Elt* dst, const Elt* src, size_t r) const { engine::binary::f3_sub_u8(dst, src, r); }
    void row_neg(Elt* dst, size_t r) const { engine::binary::f3_neg_u8(dst, r); }

    static engine::binary::TritMatrix to_trit(const RMat<Elt>& A) {
        return engine::binary::TritMatrix::from_u8(A.a.data(), A.rows, A.cols, A.cols);
    }
    HowellOut<Elt> rref_rows(engine::binary::TritMatrix& T) const {
        HowellOut<Elt> h;
        const engine::binary::TritRref rr = engine::binary::rref(T);
        h.H = RMat<Elt>(rr.rank, T.cols(), zero());
        for (uint32_t i = 0; i < rr.rank; ++i)
            for (uint32_t j = 0; j < T.cols(); ++j) h.H.at(i, j) = static_cast<Elt>(T.get(i, j));
        h.pivots = rr.pivots;
        h.pivot_v.assign(rr.rank, 0);
        return h;
    }
    HowellOut<Elt> howell(RMat<Elt> A, uint32_t k) const {
        ENGINE_CHECK_MSG(k == 1, "l23::Ops3F::howell: k = 1");
        if (A.rows == 0 || A.cols == 0) { HowellOut<Elt> h; h.H = RMat<Elt>(0, A.cols, zero()); return h; }
        engine::binary::TritMatrix T = to_trit(A);
        return rref_rows(T);
    }
    HowellOut<Elt> kernel_module(const RMat<Elt>& A, uint32_t k) const {
        ENGINE_CHECK_MSG(k == 1, "l23::Ops3F::kernel_module: k = 1");
        if (A.rows == 0 || A.cols == 0) { HowellOut<Elt> h; h.H = RMat<Elt>(0, A.cols, zero()); return h; }
        engine::binary::TritMatrix T = to_trit(A);
        const engine::binary::TritRref rr = engine::binary::rref(T);
        engine::binary::TritMatrix Kb = engine::binary::kernel_basis(T, rr.pivots);
        if (Kb.rows() == 0) { HowellOut<Elt> h; h.H = RMat<Elt>(0, A.cols, zero()); return h; }
        return rref_rows(Kb);
    }
    std::vector<uint32_t> smith(RMat<Elt> A, uint32_t k) const {
        ENGINE_CHECK_MSG(k == 1, "l23::Ops3F::smith: k = 1");
        if (A.rows == 0 || A.cols == 0) return {};
        return std::vector<uint32_t>(engine::binary::rank(to_trit(A)), 0u);
    }
};

template<class O, class = void>
struct HasF3Rows : std::false_type {};
template<class O>
struct HasF3Rows<O, std::void_t<decltype(O::kF3Rows)>> : std::integral_constant<bool, O::kF3Rows> {};
// (slot-23 R5V) Ops2 on an R5 rung (WordN<N>): its traits carry the SIMD row kernels (dst +=, -=, = - dst), which
// the Taylor shifts below use; bit-identical to the element loops (Ring2kR5SimdValidation).  Every other ring keeps
// its loop (R3 / R4 and the F_p / Z/3^k rings are untouched).
template<class O, class = void>
struct HasR5Rows : std::false_type {};
template<class O>
struct HasR5Rows<O, std::void_t<decltype(O::Tr::kR5Rows)>> : std::integral_constant<bool, O::Tr::kR5Rows> {};
// (slot-23 BIN3O1) Ops2 on a native rung of at most 64 bits (R1 ... R3: uint8_t ... uint64_t): with the BIN3O1 route on, the
// Taylor shifts' row add / sub and mat_add / mat_sub run on ring2k's dispatched AXPY kernel (r -= f p with f = -1 / +1: exact
// wraparound mod 2^W, the very same ring operation as the element loop; mat_add / mat_sub as one whole-matrix AXPY) and the Taylor
// shifts on kernel_shift_rows (a register wavefront over all columns).  R4 (u128) and R5 keep their paths.
template<class O, class = void>
struct HasNativeRows : std::false_type {};
template<class O>
struct HasNativeRows<O, std::void_t<typename O::Tr>>
    : std::integral_constant<bool, std::is_same<typename O::Tr, r2::Traits<typename O::Elt>>::value &&
                                       std::is_unsigned<typename O::Elt>::value && sizeof(typename O::Elt) <= 8 &&
                                       !HasR5Rows<O>::value> {};

// The BIN3O1 switch (see AmalgamReader.h), default on; exactly "0" = the old route, anything else warns and stays on;
// read once per process; ScopedBin3o1 overrides it.  One engine::lever::EnvLever (S24 CL-7, F10).
constexpr engine::lever::Policy kBin3o1Policy{true, engine::lever::Empty::Invalid, engine::lever::Match::Exact01,
                                              engine::lever::Cache::Once,
                                              "WARNING l23: L23_BIN3O1='%s' is neither 0 nor 1; the BIN3O1 route stays ON "
                                              "(only exactly 0 turns it off)\n"};
engine::lever::EnvArmRetiredLever g_bin3o1("L23_BIN3O1", kBin3o1Policy);
bool bin3o1_on() { return g_bin3o1.enabled(); }
std::atomic<uint64_t> g_split_accepted{0}, g_split_fallback{0}, g_odd_reduced{0}, g_rankq_blocks{0}, g_f2_cells{0}, g_hecke_reused{0};
std::atomic<uint64_t> g_rankq_fallback{0}, g_parity_fallback{0};   // (BIN3O1 audit F1) the route on, the parity structure not found

// (slot-23 BIN3O1) Ops2F: F_2 for k = 1 at ell = 2 on uint8_t entries in {0, 1} (the TER-1 Ops3F pattern).  Row kernels are
// XORs, Rho uses the characteristic-2 Taylor shift, and every elimination runs on BIN-1's M4RI binary::BitMatrix.  Its howell /
// kernel_module / smith return exactly what Ops2<uint8_t> returns at k = 1 (the RREF rows and pivots -- a Howell form over F_2 is
// the RREF --, the RREF of the D5b kernel basis, the rank): unique objects, so no value can change.
struct Ops2F {
    using Elt = uint8_t;
    static constexpr uint32_t kEll = 2;
    static constexpr bool kF2Rows = true;
    uint32_t K;
    explicit Ops2F(uint32_t K_) : K(K_) { ENGINE_CHECK_MSG(K_ == 1, "l23::Ops2F: F_2 only (K = 1)"); }
    std::string label() const { return "F_2(BitMatrix)"; }
    Elt zero() const { return 0; }
    Elt one() const { return 1; }
    Elt from_i64(int64_t v) const { return static_cast<Elt>(v & 1); }
    Elt add(Elt x, Elt y) const { return static_cast<Elt>(x ^ y); }
    Elt sub(Elt x, Elt y) const { return static_cast<Elt>(x ^ y); }
    Elt mul(Elt x, Elt y) const { return static_cast<Elt>(x & y); }
    Elt neg(Elt x) const { return x; }
    uint32_t val(Elt x, uint32_t k) const { return x ? 0 : k; }
    bool zero_mod(Elt x, uint32_t k) const { return k == 0 || x == 0; }
    Elt reduce(Elt x, uint32_t) const { return x; }
    Elt inv2() const { ENGINE_CHECK_MSG(false, "l23::Ops2F: 2 is not a unit"); return 0; }
    Elt inv_unit(Elt u) const { ENGINE_CHECK_MSG(u == 1, "l23::Ops2F::inv_unit: not a unit"); return u; }
    uint64_t lowval(Elt x, uint32_t k) const { ENGINE_CHECK_MSG(k == 1, "l23::Ops2F::lowval: k = 1"); return x; }
    // dst ^= src, eight bytes at a time (entries are 0 / 1, so XOR is addition and subtraction mod 2).
    static void xor_bytes(Elt* dst, const Elt* src, size_t r) {
        size_t c = 0;
        for (; c + 8 <= r; c += 8) {
            uint64_t a, b;
            std::memcpy(&a, dst + c, 8);
            std::memcpy(&b, src + c, 8);
            a ^= b;
            std::memcpy(dst + c, &a, 8);
        }
        for (; c < r; ++c) dst[c] = static_cast<Elt>(dst[c] ^ src[c]);
    }
    void row_add(Elt* dst, const Elt* src, size_t r) const { xor_bytes(dst, src, r); }
    void row_sub(Elt* dst, const Elt* src, size_t r) const { xor_bytes(dst, src, r); }
    void row_neg(Elt*, size_t) const {}

    static engine::binary::BitMatrix to_bits(const RMat<Elt>& A) {
        engine::binary::BitMatrix B(A.rows, A.cols);
        for (uint32_t i = 0; i < A.rows; ++i) {
            const Elt* a = A.row(i);
            uint64_t* w = B.row(i);
            for (uint32_t j = 0; j < A.cols; ++j) w[j >> 6] |= static_cast<uint64_t>(a[j] & 1u) << (j & 63);
        }
        return B;
    }
    HowellOut<Elt> rref_rows(engine::binary::BitMatrix& T) const {
        HowellOut<Elt> h;
        const engine::binary::BitRref rr = engine::binary::rref(T);
        h.H = RMat<Elt>(rr.rank, T.cols(), zero());
        for (uint32_t i = 0; i < rr.rank; ++i) {
            const uint64_t* w = T.row(i);
            Elt* o = h.H.row(i);
            for (uint32_t j = 0; j < T.cols(); ++j) o[j] = static_cast<Elt>((w[j >> 6] >> (j & 63)) & 1u);
        }
        h.pivots = rr.pivots;
        h.pivot_v.assign(rr.rank, 0);
        return h;
    }
    HowellOut<Elt> howell(RMat<Elt> A, uint32_t k) const {
        ENGINE_CHECK_MSG(k == 1, "l23::Ops2F::howell: k = 1");
        if (A.rows == 0 || A.cols == 0) { HowellOut<Elt> h; h.H = RMat<Elt>(0, A.cols, zero()); return h; }
        engine::binary::BitMatrix T = to_bits(A);
        return rref_rows(T);
    }
    HowellOut<Elt> kernel_module(const RMat<Elt>& A, uint32_t k) const {
        ENGINE_CHECK_MSG(k == 1, "l23::Ops2F::kernel_module: k = 1");
        if (A.cols == 0) { HowellOut<Elt> h; h.H = RMat<Elt>(0, 0, zero()); return h; }
        engine::binary::BitMatrix T = to_bits(A);
        const engine::binary::BitRref rr = engine::binary::rref(T);
        engine::binary::BitMatrix Kb = engine::binary::kernel_basis(T, rr.pivots);
        if (Kb.rows() == 0) { HowellOut<Elt> h; h.H = RMat<Elt>(0, A.cols, zero()); return h; }
        return rref_rows(Kb);
    }
    std::vector<uint32_t> smith(RMat<Elt> A, uint32_t k) const {
        ENGINE_CHECK_MSG(k == 1, "l23::Ops2F::smith: k = 1");
        if (A.rows == 0 || A.cols == 0) return {};
        return std::vector<uint32_t>(engine::binary::rank(to_bits(A)), 0u);
    }
};
template<class O, class = void>
struct HasF2Rows : std::false_type {};
template<class O>
struct HasF2Rows<O, std::void_t<decltype(O::kF2Rows)>> : std::integral_constant<bool, O::kF2Rows> {};

// OpsP: F_p for a large prime p < 2^31 (the rank over Q of the relation matrix; arithmetic only).
struct OpsP {
    using Elt = uint64_t;
    uint64_t p;
    uint32_t K = 0;                                                  // (BIN3O1) unused: entries are canonical, so zero_mod is == 0
    bool zero_mod(Elt x, uint32_t) const { return x == 0; }
    Elt zero() const { return 0; }
    Elt one() const { return 1; }
    Elt from_i64(int64_t v) const { return static_cast<uint64_t>(pmod(v, static_cast<int64_t>(p))); }
    Elt add(Elt x, Elt y) const { const Elt s = x + y; return s >= p ? s - p : s; }
    Elt sub(Elt x, Elt y) const { return x >= y ? x - y : x + p - y; }
    Elt mul(Elt x, Elt y) const { return x * y % p; }
    Elt neg(Elt x) const { return x ? p - x : 0; }
};

// ---------------------------------------------------------------------------
// rho_n applied structurally to a block of column vectors X (d x r; row j = coefficient of x^(n-j) y^j).
// ---------------------------------------------------------------------------
template<class Ops>
struct Rho {
    using E = typename Ops::Elt;
    using M = RMat<E>;
    const Ops& R;
    uint32_t n, d;
    bool fast;   // (slot-23 BIN3O1) the route of this object, fixed at construction
    Rho(const Ops& R_, uint32_t n_) : R(R_), n(n_), d(n_ + 1), fast(bin3o1_on()) {}

    void row_add(E* dst, const E* src, uint32_t r) const {
        if constexpr (HasF3Rows<Ops>::value || HasF2Rows<Ops>::value) R.row_add(dst, src, r);
        else if constexpr (HasR5Rows<Ops>::value) Ops::Tr::add_row(dst, src, r);
        else if constexpr (HasNativeRows<Ops>::value) {
            if (fast) { Ops::Tr::axpy(dst, src, Ops::Tr::neg(Ops::Tr::one()), r); return; }   // dst -= (-1) src
            for (uint32_t c = 0; c < r; ++c) dst[c] = R.add(dst[c], src[c]);
        }
        else for (uint32_t c = 0; c < r; ++c) dst[c] = R.add(dst[c], src[c]);
    }
    void row_sub(E* dst, const E* src, uint32_t r) const {
        if constexpr (HasF3Rows<Ops>::value || HasF2Rows<Ops>::value) R.row_sub(dst, src, r);
        else if constexpr (HasR5Rows<Ops>::value) Ops::Tr::sub_row(dst, src, r);
        else if constexpr (HasNativeRows<Ops>::value) {
            if (fast) { Ops::Tr::axpy(dst, src, Ops::Tr::one(), r); return; }                  // dst -= 1 src
            for (uint32_t c = 0; c < r; ++c) dst[c] = R.sub(dst[c], src[c]);
        }
        else for (uint32_t c = 0; c < r; ++c) dst[c] = R.sub(dst[c], src[c]);
    }
    void row_neg(E* dst, uint32_t r) const {
        if constexpr (HasF2Rows<Ops>::value) { (void)dst; (void)r; return; }   // -x = x over F_2
        else if constexpr (HasF3Rows<Ops>::value) R.row_neg(dst, r);
        else if constexpr (HasR5Rows<Ops>::value) Ops::Tr::neg_row(dst, r);
        else for (uint32_t c = 0; c < r; ++c) dst[c] = R.neg(dst[c]);
    }
    // (slot-23 TER-1) The Taylor shift p(t) -> p(t + s), s = +-1, in characteristic 3 (Ops3F only).  With M = 3^j,
    // (t + s)^M = t^M + s, so a block q0 + t^M q1 + t^(2M) q2 whose three parts are already shifted becomes
    //     s = +1:  (q0 + q1 + q2) + t^M (q1 - q2) + t^(2M) q2,     s = -1:  (q0 - q1 + q2) + t^M (q1 + q2) + t^(2M) q2;
    // levels M = 1, 3, 9, ... bottom up, rows >= d are zero (skipped).  The same linear map as T / Ti below, in
    // O(d log_3 d) row operations.
    void taylor3(M& X, bool inverse) const {
        const uint32_t r = X.cols;
        for (uint64_t Mb = 1; Mb < d; Mb *= 3) {
            for (uint64_t b = 0; b < d; b += 3 * Mb) {
                for (uint64_t u = 0; u < Mb; ++u) {
                    const uint64_t i0 = b + u, i1 = b + Mb + u, i2 = b + 2 * Mb + u;
                    if (i1 >= d) break;   // q1, q2 rows absent: nothing to combine (rows past d are zero)
                    E* r0 = X.row(static_cast<uint32_t>(i0));
                    E* r1 = X.row(static_cast<uint32_t>(i1));
                    const bool has2 = i2 < d;
                    if (!inverse) {
                        row_add(r0, r1, r);
                        if (has2) { const E* r2 = X.row(static_cast<uint32_t>(i2)); row_add(r0, r2, r); row_sub(r1, r2, r); }
                    } else {
                        row_sub(r0, r1, r);
                        if (has2) { const E* r2 = X.row(static_cast<uint32_t>(i2)); row_add(r0, r2, r); row_add(r1, r2, r); }
                    }
                }
            }
        }
    }
    // (slot-23 BIN3O1) The Taylor shift p(t) -> p(t + 1) = p(t - 1) in characteristic 2 (Ops2F only).  With M = 2^j,
    // (t + 1)^M = t^M + 1, so a block q0 + t^M q1 whose two halves are already shifted becomes (q0 + q1) + t^M q1; levels
    // M = 1, 2, 4, ... bottom up, rows >= d are zero (skipped).  The same linear map as T / Ti, in O(d log_2 d) row XORs.
    void taylor2(M& X) const {
        const uint32_t r = X.cols;
        for (uint64_t Mb = 1; Mb < d; Mb *= 2)
            for (uint64_t b = 0; b < d; b += 2 * Mb)
                for (uint64_t u = 0; u < Mb; ++u) {
                    const uint64_t i0 = b + u, i1 = b + Mb + u;
                    if (i1 >= d) break;
                    row_add(X.row(static_cast<uint32_t>(i0)), X.row(static_cast<uint32_t>(i1)), r);
                }
    }
    // (slot-23 BIN3O1) The d^2 / 2 row-operation shift over all columns of X at once (the name is historical: no column strips are
    // cut; BIN3O1 audit F3).  inverse: p(t) -> p(t - 1).  Native rungs only: row j -= f row (j + 1) (f = -1: the shift by +1,
    // f = +1: by -1, f = -a: by a), on ring2k's kernel_shift_rows (the same axpy sweep as a register wavefront of 16 passes per
    // column vector).
    void shift_strips(M& X, const E& f) const {
        if constexpr (HasNativeRows<Ops>::value) {
            if (X.cols) r2::kernel_shift_rows(X.a.data(), X.cols, n, X.cols, f);   // = the axpy sweep row j -= f row (j + 1)
        } else {
            (void)X; (void)f;
            ENGINE_CHECK_MSG(false, "l23::Rho::shift_strips: native rungs only");
        }
    }
    // (slot-23 BIN3O1) Whether shift_by has a cheap kernel for this ring (the route groups runs T^q of a word only then).  Not
    // on R5: a shift by a != +-1 is an AXPY with a multi-word product per entry there, dearer than q shifts by +-1 with R5V's
    // add / sub row kernels (measured: grouping made (2, 1000, full) 1.9x slower), so R5 keeps the letter-by-letter loop.
    static constexpr bool kShiftBy = HasF2Rows<Ops>::value || HasF3Rows<Ops>::value || HasNativeRows<Ops>::value;
    // rho(T)^a = rho(T^a): the Taylor shift p(t) -> p(t + a), row j += a row (j + 1) in the d^2 / 2 order (exact: T^a = [[1, a], [0, 1]]).
    void shift_by(M& X, int64_t a) const {
        if (a == 0) return;
        if (a == 1) { T(X); return; }
        if (a == -1) { Ti(X); return; }
        if constexpr (HasF2Rows<Ops>::value) {
            if (a & 1) taylor2(X);   // t + a = t + (a mod 2)
        } else if constexpr (HasF3Rows<Ops>::value) {
            const int64_t m = ((a % 3) + 3) % 3;
            if (m == 1) taylor3(X, false);
            else if (m == 2) taylor3(X, true);
        } else if constexpr (HasNativeRows<Ops>::value) {
            shift_strips(X, R.neg(R.from_i64(a)));
        } else {
            ENGINE_CHECK_MSG(false, "l23::Rho::shift_by: no kernel for this ring");
        }
    }
    // S = (0,-1,1,0): x^(n-j) y^j -> (-1)^j x^j y^(n-j): new[n-j] = (-1)^j old[j].
    void S(M& X) const {
        const uint32_t r = X.cols;
        for (uint32_t j = 0; 2 * j <= n; ++j) {
            const uint32_t jj = n - j;
            if (j == jj) { if (j & 1) row_neg(X.row(j), r); continue; }
            std::swap_ranges(X.row(j), X.row(j) + r, X.row(jj));   // row j <- old jj, row jj <- old j
            if (j & 1) row_neg(X.row(jj), r);                        // new[jj] = (-1)^j old[j]
            if (jj & 1) row_neg(X.row(j), r);                        // new[j] = (-1)^jj old[jj]
        }
    }
    // S^-1 = (0,1,-1,0): x^(n-j) y^j -> (-1)^(n-j) x^j y^(n-j): new[n-j] = (-1)^(n-j) old[j].
    void Sinv(M& X) const {
        const uint32_t r = X.cols;
        for (uint32_t j = 0; 2 * j <= n; ++j) {
            const uint32_t jj = n - j;
            if (j == jj) { if (jj & 1) row_neg(X.row(j), r); continue; }
            std::swap_ranges(X.row(j), X.row(j) + r, X.row(jj));
            if (jj & 1) row_neg(X.row(jj), r);                       // new[jj] = (-1)^(n-j) old[j]
            if (j & 1) row_neg(X.row(j), r);                         // new[j] = (-1)^(n-jj) old[jj]
        }
    }
    // T = (1,1,0,1): y -> x + y, the Taylor shift p(t) -> p(t + 1) in t = y / x; T^-1: p(t) -> p(t - 1).
    void T(M& X) const {
        if constexpr (HasF3Rows<Ops>::value) { taylor3(X, false); return; }
        if constexpr (HasF2Rows<Ops>::value) { taylor2(X); return; }
        if constexpr (HasNativeRows<Ops>::value) { if (fast) { shift_strips(X, Ops::Tr::neg(Ops::Tr::one())); return; } }
        const uint32_t r = X.cols;
        for (uint32_t i = 0; i < n; ++i)
            for (uint32_t j = n; j-- > i;) row_add(X.row(j), X.row(j + 1), r);
    }
    void Ti(M& X) const {
        if constexpr (HasF3Rows<Ops>::value) { taylor3(X, true); return; }
        if constexpr (HasF2Rows<Ops>::value) { taylor2(X); return; }   // t - 1 = t + 1 in characteristic 2
        if constexpr (HasNativeRows<Ops>::value) { if (fast) { shift_strips(X, Ops::Tr::one()); return; } }
        const uint32_t r = X.cols;
        for (uint32_t i = 0; i < n; ++i)
            for (uint32_t j = n; j-- > i;) row_sub(X.row(j), X.row(j + 1), r);
    }
    void U(M& X) const { T(X); S(X); }   // rho(S T) = rho(S) rho(T)
    // eps = diag(-1, 1): x^(n-j) y^j -> (-1)^(n-j) x^(n-j) y^j.
    void eps(M& X) const {
        for (uint32_t j = 0; j <= n; ++j) if ((n - j) & 1) row_neg(X.row(j), X.cols);
    }
    // diag(1, p): y -> p y.
    void D(M& X, uint32_t p) const {
        E pw = R.one();
        const E pe = R.from_i64(p);
        for (uint32_t j = 0; j <= n; ++j) {
            if (j) pw = R.mul(pw, pe);
            E* x = X.row(j);
            for (uint32_t c = 0; c < X.cols; ++c) x[c] = R.mul(x[c], pw);
        }
    }
    void letter(M& X, Letter l) const {
        if (l == Letter::S) S(X);
        else if (l == Letter::T) T(X);
        else Ti(X);
    }
    // rho(g) X for g in SL_2(Z), through its word (rho is multiplicative).
    void word(M& X, const G2& g) const {
        const std::vector<Letter> L = letters(g);
        if constexpr (kShiftBy) {
            if (fast) {   // (slot-23 BIN3O1) a run T^q (T^-q) as one shift by q (-q)
                for (size_t i = L.size(); i-- > 0;) {
                    if (L[i] == Letter::S) { S(X); continue; }
                    size_t q = 1;
                    while (i >= q && L[i - q] == L[i]) ++q;
                    shift_by(X, L[i] == Letter::T ? static_cast<int64_t>(q) : -static_cast<int64_t>(q));
                    i -= q - 1;
                }
                return;
            }
        }
        for (size_t i = L.size(); i-- > 0;) letter(X, L[i]);
    }
    M identity() const {
        M X(d, d, R.zero());
        for (uint32_t j = 0; j < d; ++j) X.at(j, j) = R.one();
        return X;
    }
};

template<class Ops>
RMat<typename Ops::Elt> mat_add(const Ops& R, RMat<typename Ops::Elt> A, const RMat<typename Ops::Elt>& B) {
    if constexpr (HasF2Rows<Ops>::value) { ENGINE_CHECK(A.a.size() == B.a.size()); R.row_add(A.a.data(), B.a.data(), A.a.size()); return A; }
    if constexpr (HasNativeRows<Ops>::value) {
        if (bin3o1_on()) {
            ENGINE_CHECK(A.a.size() == B.a.size());
            if (!A.a.empty()) Ops::Tr::axpy(A.a.data(), B.a.data(), Ops::Tr::neg(Ops::Tr::one()), A.a.size());
            return A;
        }
    }
    if constexpr (HasF3Rows<Ops>::value) { ENGINE_CHECK(A.a.size() == B.a.size()); R.row_add(A.a.data(), B.a.data(), A.a.size()); return A; }
    for (size_t i = 0; i < A.a.size(); ++i) A.a[i] = R.add(A.a[i], B.a[i]);
    return A;
}
template<class Ops>
RMat<typename Ops::Elt> mat_sub(const Ops& R, RMat<typename Ops::Elt> A, const RMat<typename Ops::Elt>& B) {
    if constexpr (HasF2Rows<Ops>::value) { ENGINE_CHECK(A.a.size() == B.a.size()); R.row_sub(A.a.data(), B.a.data(), A.a.size()); return A; }
    if constexpr (HasNativeRows<Ops>::value) {
        if (bin3o1_on()) {
            ENGINE_CHECK(A.a.size() == B.a.size());
            if (!A.a.empty()) Ops::Tr::axpy(A.a.data(), B.a.data(), Ops::Tr::one(), A.a.size());
            return A;
        }
    }
    if constexpr (HasF3Rows<Ops>::value) { ENGINE_CHECK(A.a.size() == B.a.size()); R.row_sub(A.a.data(), B.a.data(), A.a.size()); return A; }
    for (size_t i = 0; i < A.a.size(); ++i) A.a[i] = R.sub(A.a[i], B.a[i]);
    return A;
}
template<class Ops>
RMat<typename Ops::Elt> mat_mul(const Ops& R, const RMat<typename Ops::Elt>& A, const RMat<typename Ops::Elt>& B) {
    ENGINE_CHECK(A.cols == B.rows);
    RMat<typename Ops::Elt> C(A.rows, B.cols, R.zero());
    for (uint32_t i = 0; i < A.rows; ++i) {
        auto* ci = C.row(i);
        for (uint32_t t = 0; t < A.cols; ++t) {
            const auto& f = A.at(i, t);
            if (R.zero_mod(f, R.K)) continue;
            const auto* bt = B.row(t);
            for (uint32_t j = 0; j < B.cols; ++j) ci[j] = R.add(ci[j], R.mul(f, bt[j]));
        }
    }
    return C;
}
template<class Ops>
RMat<typename Ops::Elt> transpose(const RMat<typename Ops::Elt>& A, const Ops& R) {
    RMat<typename Ops::Elt> T(A.cols, A.rows, R.zero());
    for (uint32_t i = 0; i < A.rows; ++i)
        for (uint32_t j = 0; j < A.cols; ++j) T.at(j, i) = A.at(i, j);
    return T;
}
template<class Ops>
RMat<typename Ops::Elt> scalar(const Ops& R, uint32_t r, const typename Ops::Elt& c) {
    RMat<typename Ops::Elt> X(r, r, R.zero());
    for (uint32_t i = 0; i < r; ++i) X.at(i, i) = c;
    return X;
}
template<class Ops>
bool all_zero_mod(const Ops& R, const RMat<typename Ops::Elt>& A, uint32_t k) {
    for (const auto& x : A.a) if (!R.zero_mod(x, k)) return false;
    return true;
}
template<class Ops>
typename Ops::Elt power(const Ops& R, uint64_t b, uint32_t e) {
    typename Ops::Elt x = R.one();
    const typename Ops::Elt be = R.from_i64(static_cast<int64_t>(b));
    for (uint32_t i = 0; i < e; ++i) x = R.mul(x, be);
    return x;
}

// ---------------------------------------------------------------------------
// The amalgam: relation matrix A (3d x 2d), coboundary generators B (2d x d), cocycle evaluation, T_p, iota.
// ---------------------------------------------------------------------------
template<class Ops>
struct Amalgam {
    using E = typename Ops::Elt;
    using M = RMat<E>;
    const Ops& R;
    Rho<Ops> rho;
    uint32_t n, d;
    uint32_t mutate = 0;
    Amalgam(const Ops& R_, uint32_t n_, uint32_t mutate_ = 0) : R(R_), rho(R_, n_), n(n_), d(n_ + 1), mutate(mutate_) {}

    // The four d x d blocks of the relation matrix: N_S = 1 + S + S^2 + S^3, N_U = 1 + U + ... + U^5, 1 + S, 1 + U + U^2.
    struct Blocks { M NS, NU, onepS, onepU; };
    // (slot-23 BIN3O1) rho(T) on the identity is the Pascal matrix P[i][j] = C(j, i) (column j = (x + y)^j's coefficients: row i
    // = the coefficient of x^(n-i) y^i), built by additions in O(d^2).
    M pascal() const {
        M P(d, d, R.zero());
        for (uint32_t j = 0; j < d; ++j) P.at(0, j) = R.one();
        for (uint32_t i = 1; i < d; ++i)
            for (uint32_t j = i; j < d; ++j) P.at(i, j) = R.add(P.at(i, j - 1), P.at(i - 1, j - 1));   // C(j,i) = C(j-1,i) + C(j-1,i-1)
        return P;
    }
    // (slot-23 BIN3O1) rho(U)^k, k = 1 .. 5, in O(d^2) each: rho(U) = rho(S) P (a signed row permutation of P);
    // rho(U)^2 = rho(U^-1 U^3) with U^3 = -I, rho(U^-1) = rho(T^-1) rho(S^-1): U2[i][j] = (-1)^(n-i) C(n-j, i) = (-1)^(n-i) P[i][n-j];
    // rho(U)^3 = rho(-I) = (-1)^n I, rho(U)^4 = (-1)^n rho(U), rho(U)^5 = (-1)^n rho(U)^2.  Exact identities over every ring.
    std::vector<M> upowers_fast() const {
        std::vector<M> Up(6);
        Up[0] = rho.identity();
        const M P = pascal();
        Up[1] = P;
        rho.S(Up[1]);
        Up[2] = M(d, d, R.zero());
        for (uint32_t i = 0; i < d; ++i)
            for (uint32_t j = 0; j < d; ++j) {
                const auto& c = P.at(i, n - j);
                Up[2].at(i, j) = ((n - i) & 1) ? R.neg(c) : c;
            }
        const bool odd = n & 1;
        Up[3] = Up[0];
        Up[4] = Up[1];
        Up[5] = Up[2];
        if (odd) for (int k = 3; k <= 5; ++k) for (auto& x : Up[k].a) x = R.neg(x);
        return Up;
    }
    Blocks blocks() const {
        const M I = rho.identity();
        Blocks b;
        M X = I;
        b.NS = I;
        for (int i = 0; i < 3; ++i) { rho.S(X); b.NS = mat_add(R, std::move(b.NS), X); }
        b.onepS = I;
        { M s = I; rho.S(s); b.onepS = mat_add(R, std::move(b.onepS), s); }
        b.NU = I;
        b.onepU = I;
        if (rho.fast) {
            const std::vector<M> Up = upowers_fast();
            for (int i = 1; i <= 5; ++i) {
                b.NU = mat_add(R, std::move(b.NU), Up[i]);
                if (i <= 2) b.onepU = mat_add(R, std::move(b.onepU), Up[i]);
            }
            return b;
        }
        M Y = I;
        for (int i = 1; i <= 5; ++i) {
            rho.U(Y);
            b.NU = mat_add(R, std::move(b.NU), Y);
            if (i <= 2) b.onepU = mat_add(R, std::move(b.onepU), Y);
        }
        return b;
    }
    M assemble(const Blocks& b) const {
        M A(3 * d, 2 * d, R.zero());
        for (uint32_t i = 0; i < d; ++i)
            for (uint32_t j = 0; j < d; ++j) {
                A.at(i, j) = b.NS.at(i, j);
                A.at(d + i, d + j) = b.NU.at(i, j);
                A.at(2 * d + i, j) = b.onepS.at(i, j);
                A.at(2 * d + i, d + j) = R.neg(b.onepU.at(i, j));
            }
        return A;
    }
    M relation_matrix() const { return assemble(blocks()); }
    // (slot-23 BIN3O1) The structure of A by the parity of n, CHECKED entrywise (O(d^2)): rho(-I) = (-1)^n, so S^2 = U^3 = (-1)^n
    // and exactly N_S = (1 + (-1)^n)(1 + S), N_U = (1 + (-1)^n)(1 + U + U^2).  true iff it holds (it always does; a false
    // return sends the caller to the old route).
    bool parity_structure(const Blocks& b) const {
        const bool even = (n % 2) == 0;
        for (size_t t = 0; t < b.NS.a.size(); ++t) {
            const auto ws = even ? R.add(b.onepS.a[t], b.onepS.a[t]) : R.zero();
            const auto wu = even ? R.add(b.onepU.a[t], b.onepU.a[t]) : R.zero();
            if (!R.zero_mod(R.sub(b.NS.a[t], ws), R.K) || !R.zero_mod(R.sub(b.NU.a[t], wu), R.K)) return false;
        }
        return true;
    }
    // [1 + S | -(1 + U + U^2)]: the third block row of A (d x 2d).
    M third_row(const Blocks& b) const {
        M A(d, 2 * d, R.zero());
        for (uint32_t i = 0; i < d; ++i)
            for (uint32_t j = 0; j < d; ++j) {
                A.at(i, j) = b.onepS.at(i, j);
                A.at(i, d + j) = R.neg(b.onepU.at(i, j));
            }
        return A;
    }
    M coboundaries() const {   // [(S - 1); (U - 1)], 2d x d
        const M I = rho.identity();
        M s = I, u = I;
        rho.S(s);
        if (rho.fast) { u = pascal(); rho.S(u); }   // (slot-23 BIN3O1) rho(U) = rho(S) P
        else rho.U(u);
        s = mat_sub(R, std::move(s), I);
        u = mat_sub(R, std::move(u), I);
        M B(2 * d, d, R.zero());
        for (uint32_t i = 0; i < d; ++i)
            for (uint32_t j = 0; j < d; ++j) { B.at(i, j) = s.at(i, j); B.at(d + i, j) = u.at(i, j); }
        return B;
    }
    // Rows [r0, r0 + d) of a 2d x r block.
    M half(const M& C, uint32_t r0) const {
        M X(d, C.cols, R.zero());
        std::copy(C.row(r0), C.row(r0) + static_cast<size_t>(d) * C.cols, X.a.begin());
        return X;
    }
    M stack(const M& top, const M& bot) const {
        M X(2 * d, top.cols, R.zero());
        std::copy(top.a.begin(), top.a.end(), X.a.begin());
        std::copy(bot.a.begin(), bot.a.end(), X.a.begin() + static_cast<std::ptrdiff_t>(top.a.size()));
        return X;
    }
    // The cocycles' values on the letters: c(S), c(T) = c(S^-1) + S^-1 c(U), c(T^-1) = -T^-1 c(T), c(S^-1) = -S^-1 c(S).
    struct Vals { M cS, cU, cSinv, cT, cTi; };
    Vals values(const M& C) const {
        Vals v;
        v.cS = half(C, 0);
        v.cU = half(C, d);
        v.cSinv = v.cS;
        rho.Sinv(v.cSinv);
        for (auto& x : v.cSinv.a) x = R.neg(x);
        M t = v.cU;
        rho.Sinv(t);
        v.cT = mat_add(R, v.cSinv, t);
        v.cTi = v.cT;
        rho.Ti(v.cTi);
        for (auto& x : v.cTi.a) x = R.neg(x);
        return v;
    }
    // c(g) by Horner along the word, right to left: c(s1 ... sm) = c(s1) + s1 c(s2 ... sm).
    M ev(const Vals& v, const G2& g) const {
        const std::vector<Letter> L = letters(g);
        M val(d, v.cS.cols, R.zero());
        if constexpr (Rho<Ops>::kShiftBy) {
            if (rho.fast) {
                // (slot-23 BIN3O1) A run X^q (X = T or T^-1, q >= 4) in one step: val <- c(X^q) + X^q val, with c(X^q) = sum_{j<q} X^j c(X)
                // by doubling (c(X^(a+b)) = c(X^a) + X^a c(X^b)): O(log q) shifts by multiples instead of q shifts by 1.  The same
                // ring element (exact identities); shorter runs and S go letter by letter as before.
                for (size_t i = L.size(); i-- > 0;) {
                    size_t q = 1;
                    if (L[i] != Letter::S) while (i >= q && L[i - q] == L[i]) ++q;
                    if (q < 4) {
                        for (size_t t = 0; t < q; ++t) {
                            const Letter l = L[i - t];
                            rho.letter(val, l);
                            const M& c = l == Letter::S ? v.cS : (l == Letter::T ? v.cT : v.cTi);
                            val = mat_add(R, std::move(val), c);
                        }
                        i -= q - 1;
                        continue;
                    }
                    const int64_t sgn = L[i] == Letter::T ? 1 : -1;
                    M acc, B = L[i] == Letter::T ? v.cT : v.cTi;   // B = c(X^b), acc = c(X^e)
                    bool has = false;
                    uint64_t e = 0, b = 1, qq = q;
                    while (qq) {
                        if (qq & 1) {
                            M t = B;
                            rho.shift_by(t, sgn * static_cast<int64_t>(e));
                            acc = has ? mat_add(R, std::move(acc), t) : std::move(t);
                            has = true;
                            e += b;
                        }
                        qq >>= 1;
                        if (qq) {
                            M t = B;
                            rho.shift_by(t, sgn * static_cast<int64_t>(b));
                            B = mat_add(R, std::move(B), t);
                            b *= 2;
                        }
                    }
                    rho.shift_by(val, sgn * static_cast<int64_t>(q));
                    val = mat_add(R, std::move(val), acc);
                    i -= q - 1;
                }
                return val;
            }
        }
        for (size_t i = L.size(); i-- > 0;) {
            rho.letter(val, L[i]);
            const M& c = L[i] == Letter::S ? v.cS : (L[i] == Letter::T ? v.cT : v.cTi);
            val = mat_add(R, std::move(val), c);
        }
        return val;
    }
    // T_p on the cocycles in the columns of C (2d x r): (T_p c)(gen) = sum_r rho(r^-1 g) c(g^-1 h_r g).
    M hecke(uint32_t p, const M& C) const {
        const Vals v = values(C);
        const std::vector<G2> reps = coset_reps(p);
        const G2 gens[2] = {kS, kU};
        M out[2];
        for (int gi = 0; gi < 2; ++gi) {
            M acc(d, C.cols, R.zero());
            for (uint32_t ri = 0; ri < reps.size(); ++ri) {
                if (mutate == 1 && ri + 1 == reps.size()) continue;   // NEGATIVE CONTROL: drop r_inf
                const G2 rg = gmul(reps[ri], gens[gi]);
                const uint32_t j = coset_index(rg, p);
                const G2 h = gmul(rg, ginv(reps[j]));
                ENGINE_CHECK_MSG(h.c % static_cast<int64_t>(p) == 0, "l23::hecke: h not in Gamma_0(p)");
                const G2 ghg{h.a, h.b * static_cast<int64_t>(p), h.c / static_cast<int64_t>(p), h.d};
                M x = ev(v, ghg);
                rho.D(x, p);                  // rho(r^-1 g) = rho(r^-1) rho(diag(1, p))
                rho.word(x, ginv(reps[ri]));
                acc = mat_add(R, std::move(acc), x);
            }
            out[gi] = std::move(acc);
        }
        return stack(out[0], out[1]);
    }
    // iota c := -rho(eps) c(eps . eps): (iota c)(S) = -eps c(S^-1), (iota c)(U) = -eps c(eps U eps), eps U eps = (0,1,-1,1).
    M iota(const M& C) const {
        const Vals v = values(C);
        M a = v.cSinv;
        M b = ev(v, G2{0, 1, -1, 1});
        rho.eps(a);
        rho.eps(b);
        for (auto& x : a.a) x = R.neg(x);
        for (auto& x : b.a) x = R.neg(x);
        return stack(a, b);
    }
    // The relator conditions applied structurally: zero mod ell^k iff the columns are cocycles mod ell^k.
    bool cocycles(const M& C, uint32_t k) const {
        const M cS = half(C, 0), cU = half(C, d);
        M x = cS, ns = cS;
        for (int i = 0; i < 3; ++i) { rho.S(x); ns = mat_add(R, std::move(ns), x); }
        if (!all_zero_mod(R, ns, k)) return false;
        M y = cU, nu = cU, pu = cU;
        for (int i = 1; i <= 5; ++i) {
            rho.U(y);
            nu = mat_add(R, std::move(nu), y);
            if (i <= 2) pu = mat_add(R, std::move(pu), y);
        }
        if (!all_zero_mod(R, nu, k)) return false;
        M s1 = cS;
        rho.S(s1);
        s1 = mat_add(R, std::move(s1), cS);
        return all_zero_mod(R, mat_sub(R, std::move(s1), pu), k);
    }
};

// The unit-pivot reduced basis of a FREE DIRECT SUMMAND N of (Z/ell^k)^n, given generating rows G: Gauss-Jordan
// with unit pivots only, columns left to right (the pivot columns are the F_ell column rank profile of N mod ell),
// each pivot scaled to 1 and cleared above and below.  The result is unique: rows b_1 .. b_z with b_i = e_i on
// the pivot columns, so w = sum_i w[pivot_i] b_i for every w in N.  The Howell form is NOT such a basis (a free
// summand such as Z/4 (2, 1) has a non-unit leading entry).  ok = false unless every non-pivot row reduces to
// zero (i.e. N was free of rank z = #pivots and a direct summand).
template<class E>
struct UnitBasis {
    RMat<E> B;
    std::vector<uint32_t> pivots;
    bool ok = false;
};
template<class Ops>
UnitBasis<typename Ops::Elt> unit_basis(const Ops& R, RMat<typename Ops::Elt> G, uint32_t k) {
    using E = typename Ops::Elt;
    UnitBasis<E> out;
    const uint32_t g = G.rows, n = G.cols;
    uint32_t r = 0;
    for (uint32_t c = 0; c < n && r < g; ++c) {
        uint32_t piv = g;
        for (uint32_t i = r; i < g; ++i) if (R.val(G.at(i, c), k) == 0) { piv = i; break; }
        if (piv == g) continue;
        if (piv != r) std::swap_ranges(G.row(piv), G.row(piv) + n, G.row(r));
        E* pr = G.row(r);
        const E u = R.inv_unit(pr[c]);
        for (uint32_t j = 0; j < n; ++j) pr[j] = R.reduce(R.mul(pr[j], u), k);
        for (uint32_t i = 0; i < g; ++i) {
            if (i == r) continue;
            E* ri = G.row(i);
            const E f = R.reduce(ri[c], k);
            if (R.zero_mod(f, k)) continue;
            for (uint32_t j = 0; j < n; ++j) ri[j] = R.reduce(R.sub(ri[j], R.mul(f, pr[j])), k);
        }
        out.pivots.push_back(c);
        ++r;
    }
    out.ok = true;
    for (uint32_t i = r; i < g && out.ok; ++i)
        for (uint32_t j = 0; j < n; ++j) if (!R.zero_mod(G.at(i, j), k)) { out.ok = false; break; }
    out.B = RMat<E>(r, n, R.zero());
    std::copy(G.a.begin(), G.a.begin() + static_cast<std::ptrdiff_t>(static_cast<size_t>(r) * n), out.B.a.begin());
    return out;
}

// Columns of a Howell form's rows (a generating set as a 2d x h block).
template<class Ops>
RMat<typename Ops::Elt> rows_as_columns(const Ops& R, const HowellOut<typename Ops::Elt>& h, uint32_t len) {
    RMat<typename Ops::Elt> X(len, h.H.rows, R.zero());
    for (uint32_t i = 0; i < h.H.rows; ++i)
        for (uint32_t j = 0; j < len; ++j) X.at(j, i) = h.H.at(i, j);
    return X;
}
// log_ell of the order of the Z/ell^k-span of the columns of X.
template<class Ops>
uint32_t span_length(const Ops& R, const RMat<typename Ops::Elt>& X, uint32_t k) {
    if (X.cols == 0 || X.rows == 0) return 0;
    const std::vector<uint32_t> s = R.smith(transpose(X, R), k);
    uint32_t L = 0;
    for (uint32_t x : s) L += k - x;
    return L;
}
template<class Ops>
RMat<typename Ops::Elt> hcat(const Ops& R, const RMat<typename Ops::Elt>& X, const RMat<typename Ops::Elt>& Y) {
    ENGINE_CHECK(X.rows == Y.rows);
    RMat<typename Ops::Elt> Z(X.rows, X.cols + Y.cols, R.zero());
    for (uint32_t i = 0; i < X.rows; ++i) {
        std::copy(X.row(i), X.row(i) + X.cols, Z.row(i));
        std::copy(Y.row(i), Y.row(i) + Y.cols, Z.row(i) + X.cols);
    }
    return Z;
}
// log_ell |(span X + span B) / span B|
template<class Ops>
uint32_t quotient_length(const Ops& R, const RMat<typename Ops::Elt>& X, const RMat<typename Ops::Elt>& B, uint32_t k) {
    return span_length(R, hcat(R, X, B), k) - span_length(R, B, k);
}

// The rank over Q of the relation matrix: the largest rank modulo three primes near 2^30 (an upper bound
// is the rank over Q; a prime dividing every maximal non-zero minor would have to divide three fixed integers).
uint32_t rank_fp(uint64_t p, const RMat<uint64_t>& A) {
    if (A.rows == 0 || A.cols == 0) return 0;
    engine::DenseBuf<uint64_t> b(A.rows, A.cols);
    for (uint32_t i = 0; i < A.rows; ++i)
        for (uint32_t j = 0; j < A.cols; ++j) b.at(i, j) = A.at(i, j);
    const engine::ModCtx<uint64_t> F(p);
    return engine::elim::column_rank_profile(F, engine::DenseView<uint64_t>(b)).rank;
}
uint32_t rank_q_relation(uint32_t n) {
    uint32_t best = 0;
    for (uint64_t p : {2147483647ull, 1000000007ull, 998244353ull}) {
        const OpsP R{p};
        const Amalgam<OpsP> am(R, n);
        uint32_t r = 0;
        const typename Amalgam<OpsP>::Blocks bl = am.blocks();
        const bool par = am.rho.fast && am.parity_structure(bl);
        if (am.rho.fast && !par) g_rankq_fallback.fetch_add(1, std::memory_order_relaxed);   // (BIN3O1 audit F1)
        if (par) {
            // (slot-23 BIN3O1) p is odd, so 2 is a unit: n even, A = [2X 0; 0 2Y; X -Y] has the row space of [X 0; 0 Y] and
            // rank A = rank X + rank Y (X = 1 + S, Y = 1 + U + U^2, two d x d); n odd, A = [0 0; 0 0; X -Y] and rank A = rank [X | -Y].
            g_rankq_blocks.fetch_add(1, std::memory_order_relaxed);
            if (n % 2 == 0) r = rank_fp(p, bl.onepS) + rank_fp(p, bl.onepU);
            else r = rank_fp(p, am.third_row(bl));
        } else {
            r = rank_fp(p, am.assemble(bl));
        }
        best = std::max(best, r);
    }
    return best;
}

template<class Ops>
Divisors make_divisors(const std::vector<uint32_t>& sigma, uint32_t rank, uint32_t cap) {
    Divisors D;
    D.cap = cap;
    for (uint32_t s : sigma) if (s < cap) D.finite.push_back(s);
    std::sort(D.finite.begin(), D.finite.end());
    D.at_least = rank - static_cast<uint32_t>(D.finite.size());
    return D;
}

// ---------------------------------------------------------------------------
// One attempt of a lattice cell at working precision R.K.
// ---------------------------------------------------------------------------
template<class Ops>
void lattice_attempt(const Ops& R, uint32_t n, const ReaderOptions& opt, uint32_t cap_req, LatticeCell& out) {
    using E = typename Ops::Elt;
    using M = RMat<E>;
    const uint32_t ell = Ops::kEll, K = R.K, d = n + 1;
    out.failures.clear();
    out.ok = false;
    out.K = K;
    out.rung = R.label();
    auto fail = [&](const std::string& s) { out.failures.push_back(s); };
    const Amalgam<Ops> am(R, n, opt.mutate);
    typename Amalgam<Ops>::Blocks bl = am.blocks();
    const M A = am.assemble(bl);
    const M Bg = am.coboundaries();
    // h0 = dim H^0(Gamma; M) = d - rank_F_ell [S - 1; U - 1]
    out.h0 = d - static_cast<uint32_t>(R.smith(Bg, 1).size());
    // (1) sigma(A) with the rank over Q as the hint.
    bool verified = false;
    const std::vector<uint32_t> sA = R.smith_hinted(A, out.rank_q_A, out.log, verified);
    if (!verified) { fail("sigma(A): the rank over Q is not met at K (an invariant >= K)"); return; }
    out.e_A = sA.empty() ? 0 : sA.back();
    const bool guard = opt.mutate != 2;   // mutate 2 = NEGATIVE CONTROL: no precision guard
    const uint32_t m1 = guard ? K - out.e_A : K;
    if (m1 < 2) { fail("m1 < 2"); return; }
    // (2) the saturated cocycle lattice L = red_{m1} ker(A mod ell^K).
    out.z = 2 * d - out.rank_q_A;
    auto lattice_old = [&]() {
        HowellOut<E> Zk = R.kernel_module(A, K);
        M Zr = Zk.H;
        for (auto& x : Zr.a) x = R.reduce(x, m1);
        return unit_basis(R, std::move(Zr), m1);
    };
    UnitBasis<E> Lh;
    bool have_L = false;
    // (slot-23 BIN3O1) The parity structure of A (Amalgam::parity_structure; not under a negative control, whose old-route
    // failure text must stay what it is).  BIN3O1 audit F2: the acceptance below (G contains red_{m1} L) relies on the split reading
    // the very blocks `bl` that built A above (assemble(bl)); never feed it blocks computed separately.
    const bool par = am.rho.fast && opt.mutate == 0 && am.parity_structure(bl);
    if (am.rho.fast && opt.mutate == 0 && !par) g_parity_fallback.fetch_add(1, std::memory_order_relaxed);   // (BIN3O1 audit F1)
    if (par) {
        if (n % 2 == 1) {
            // n odd: the first 2d rows of A are exactly zero, so ker(A mod ell^K) = ker([1 + S | -(1 + U + U^2)] mod ell^K): the same
            // module, hence the same Howell form and the same unit-pivot basis.
            g_odd_reduced.fetch_add(1, std::memory_order_relaxed);
            HowellOut<E> Zk = R.kernel_module(am.third_row(bl), K);
            M Zr = Zk.H;
            for (auto& x : Zr.a) x = R.reduce(x, m1);
            Lh = unit_basis(R, std::move(Zr), m1);
            have_L = true;
        } else {
            // n even: over Z, A (c_S, c_U) = 0 iff (1 + S) c_S = 0 and (1 + U + U^2) c_U = 0 (A = [2X 0; 0 2Y; X -Y]), so
            // L = (ker_Z X x ker_Z Y) (x) Z_ell.  G = red_{m1} of ker(X mod ell^K) x ker(Y mod ell^K) contains red_{m1} L; if
            // unit_basis finds G free of rank z, G is a free summand of the rank of the free summand red_{m1} L inside it, so G =
            // red_{m1} L (the old route's module), and its unit-pivot basis is unique: the same rows and pivots.  Otherwise the
            // old route runs (counted).
            const HowellOut<E> KX = R.kernel_module(bl.onepS, K);
            const HowellOut<E> KY = R.kernel_module(bl.onepU, K);
            M G(KX.H.rows + KY.H.rows, 2 * d, R.zero());
            for (uint32_t i = 0; i < KX.H.rows; ++i)
                for (uint32_t j = 0; j < d; ++j) G.at(i, j) = R.reduce(KX.H.at(i, j), m1);
            for (uint32_t i = 0; i < KY.H.rows; ++i)
                for (uint32_t j = 0; j < d; ++j) G.at(KX.H.rows + i, d + j) = R.reduce(KY.H.at(i, j), m1);
            Lh = unit_basis(R, std::move(G), m1);
            have_L = Lh.ok && Lh.B.rows == out.z;
            (have_L ? g_split_accepted : g_split_fallback).fetch_add(1, std::memory_order_relaxed);
        }
    }
    bl = typename Amalgam<Ops>::Blocks();   // the blocks are not needed past here
    if (!have_L) Lh = lattice_old();
    if (Lh.B.rows != out.z || !Lh.ok) {
        fail("L: " + std::to_string(Lh.B.rows) + " unit pivots (want z = " + std::to_string(out.z) + "), free summand " +
             (Lh.ok ? "yes" : "NO"));
        return;
    }
    const std::vector<uint32_t>& pz = Lh.pivots;
    // (3) C = coordinates of B^1 (z x d); sigma(C) = the torsion of H^1(Gamma; V).
    M C(out.z, d, R.zero());
    for (uint32_t i = 0; i < out.z; ++i)
        for (uint32_t j = 0; j < d; ++j) C.at(i, j) = R.reduce(Bg.at(pz[i], j), m1);
    // B^1 lies in L: every coboundary equals its coordinate combination of the L rows (mod ell^m1).
    {
        const M recon = mat_mul(R, transpose(Lh.B, R), C);   // 2d x d
        bool inL = true;
        for (size_t t = 0; t < recon.a.size() && inL; ++t) inL = R.zero_mod(R.sub(recon.a[t], Bg.a[t]), m1);
        if (!inL) { fail("B^1 not in the span of L"); return; }
    }
    const std::vector<uint32_t> sC = R.smith(C, m1);
    if (sC.size() != d) { fail("sigma(C): " + std::to_string(sC.size()) + " invariants (want d = rank B^1)"); return; }
    out.e_1 = sC.empty() ? 0 : sC.back();
    out.torsion.clear();
    for (uint32_t s : sC) if (s > 0) out.torsion.push_back(s);
    std::sort(out.torsion.begin(), out.torsion.end(), [](uint32_t a, uint32_t b) { return a > b; });
    const uint32_t m2 = guard ? m1 - out.e_1 : m1;
    if (m2 < 1) { fail("m < 1"); return; }
    out.m = m2;
    // (4) Lambda^1's dual: the left kernel of C, reduced to m2.
    out.rank_lambda = out.z - d;
    HowellOut<E> Fk = R.kernel_module(transpose(C, R), m1);
    M Fr = Fk.H;
    for (auto& x : Fr.a) x = R.reduce(x, m2);
    const UnitBasis<E> Fh = unit_basis(R, std::move(Fr), m2);
    if (Fh.B.rows != out.rank_lambda || !Fh.ok) {
        fail("Lambda^dual: " + std::to_string(Fh.B.rows) + " unit pivots (want " + std::to_string(out.rank_lambda) + "), free summand " +
             (Fh.ok ? "yes" : "NO"));
        return;
    }
    const uint32_t rk = out.rank_lambda, m = m2;
    const uint32_t cap = cap_req ? cap_req : m;
    if (cap > m) { fail("certified m = " + std::to_string(m) + " < the report cap " + std::to_string(cap)); return; }
    // The dual basis lambda_j: the L rows at F's pivot columns (2d x rk cocycles).
    M Lam(2 * d, rk, R.zero());
    for (uint32_t j = 0; j < rk; ++j)
        for (uint32_t t = 0; t < 2 * d; ++t) Lam.at(t, j) = Lh.B.at(Fh.pivots[j], t);
    auto proj = [&](const M& Wc) {   // rk x r: f_i(w) = sum_t F[i][t] w[pz[t]]
        M Y(rk, Wc.cols, R.zero());
        for (uint32_t i = 0; i < rk; ++i)
            for (uint32_t t = 0; t < out.z; ++t) {
                const E& f = Fh.B.at(i, t);
                if (R.zero_mod(f, m)) continue;
                const E* w = Wc.row(pz[t]);
                E* y = Y.row(i);
                for (uint32_t c = 0; c < Wc.cols; ++c) y[c] = R.add(y[c], R.mul(f, w[c]));
            }
        for (auto& x : Y.a) x = R.reduce(x, m);
        return Y;
    };
    // (5) the Hecke operators on Lambda^1.
    std::map<uint32_t, M> T;
    for (uint32_t p : opt.primes) {
        const M TL = am.hecke(p, Lam);
        if (!am.cocycles(TL, m1)) fail("T_" + std::to_string(p) + " Lambda: not cocycles mod ell^m1");
        T[p] = proj(TL);
    }
    out.commute = true;
    for (auto it = T.begin(); it != T.end(); ++it)
        for (auto jt = std::next(it); jt != T.end(); ++jt)
            if (!all_zero_mod(R, mat_sub(R, mat_mul(R, it->second, jt->second), mat_mul(R, jt->second, it->second)), m))
                out.commute = false;
    if (!out.commute) fail("the T_p do not commute mod ell^m");
    out.ed_tp.clear();
    out.ed_eis.clear();
    for (const auto& kv : T) {
        const uint32_t p = kv.first;
        if (p == ell) continue;
        const M tp = mat_sub(R, kv.second, scalar(R, rk, R.from_i64(1 + p)));
        const M te = mat_sub(R, kv.second, scalar(R, rk, R.add(R.one(), power(R, p, n + 1))));
        out.ed_tp[p] = make_divisors<Ops>(R.smith(tp, m), rk, cap);
        out.ed_eis[p] = make_divisors<Ops>(R.smith(te, m), rk, cap);
    }
    const auto tl = T.find(ell);
    if (tl != T.end()) {
        out.ed_tell = make_divisors<Ops>(R.smith(tl->second, m), rk, cap);
        out.ed_tell2 = make_divisors<Ops>(R.smith(mat_sub(R, mat_mul(R, tl->second, tl->second), tl->second), m), rk, cap);
    }
    // (6) ell = 3: iota, Lambda^+-, ed_*_pm.
    out.pm_done = false;
    if (ell == 3 && opt.pm) {
        const M IL = am.iota(Lam);
        if (!am.cocycles(IL, m1)) fail("iota Lambda: not cocycles");
        const M J = proj(IL);
        const M I = scalar(R, rk, R.one());
        out.iota_ok = all_zero_mod(R, mat_sub(R, mat_mul(R, J, J), I), m);
        for (const auto& kv : T)
            out.iota_ok = out.iota_ok && all_zero_mod(R, mat_sub(R, mat_mul(R, J, kv.second), mat_mul(R, kv.second, J)), m);
        if (!out.iota_ok) fail("iota: J^2 != 1 or J does not commute with the T_p");
        const E h = R.inv2();
        M Pp = mat_add(R, I, J), Pm = mat_sub(R, I, J);
        for (auto& x : Pp.a) x = R.reduce(R.mul(x, h), m);
        for (auto& x : Pm.a) x = R.reduce(R.mul(x, h), m);
        const std::vector<uint32_t> sp = R.smith(Pp, m), sm = R.smith(Pm, m);
        out.dim_plus = static_cast<uint32_t>(std::count(sp.begin(), sp.end(), 0u));
        out.dim_minus = static_cast<uint32_t>(std::count(sm.begin(), sm.end(), 0u));
        if (sp.size() != out.dim_plus || sm.size() != out.dim_minus || out.dim_plus + out.dim_minus != rk)
            fail("Lambda^+- : the projectors are not complementary idempotents");
        auto restricted = [&](const M& X, bool plus, Divisors& D) {
            // ed(X | Lambda^+) = ed(X P+ + P-) minus dim Lambda^- units (and symmetrically).
            const M Y = mat_add(R, mat_mul(R, X, plus ? Pp : Pm), plus ? Pm : Pp);
            Divisors full = make_divisors<Ops>(R.smith(Y, m), rk, cap);
            uint32_t drop = plus ? out.dim_minus : out.dim_plus;
            Divisors r;
            r.cap = cap;
            r.at_least = full.at_least;
            for (uint32_t x : full.finite) {
                if (x == 0 && drop) { --drop; continue; }
                r.finite.push_back(x);
            }
            if (drop) fail("ed_pm: fewer units than dim Lambda^-+");
            D = r;
        };
        out.ed_tp_plus.clear(); out.ed_tp_minus.clear(); out.ed_eis_plus.clear(); out.ed_eis_minus.clear();
        for (const auto& kv : T) {
            const uint32_t p = kv.first;
            if (p == ell) continue;
            const M tp = mat_sub(R, kv.second, scalar(R, rk, R.from_i64(1 + p)));
            const M te = mat_sub(R, kv.second, scalar(R, rk, R.add(R.one(), power(R, p, n + 1))));
            restricted(tp, true, out.ed_tp_plus[p]);
            restricted(tp, false, out.ed_tp_minus[p]);
            restricted(te, true, out.ed_eis_plus[p]);
            restricted(te, false, out.ed_eis_minus[p]);
        }
        if (tl != T.end()) {
            restricted(tl->second, true, out.ed_tell_plus);
            restricted(tl->second, false, out.ed_tell_minus);
        }
        out.pm_done = true;
    }
    out.ok = out.failures.empty();
}

// ---------------------------------------------------------------------------
// Rung dispatch
// ---------------------------------------------------------------------------
template<class F>
void with_ops2(uint32_t K, F&& f) {
    const r2::RungChoice c = r2::select_rung(K < 2 ? 2 : K);
    switch (c.rung) {
    case r2::Rung::R0:
    case r2::Rung::R1: { Ops2<uint8_t> R(K); f(R); return; }
    case r2::Rung::R2_16: { Ops2<uint16_t> R(K); f(R); return; }
    case r2::Rung::R2_32: { Ops2<uint32_t> R(K); f(R); return; }
    case r2::Rung::R3: { Ops2<uint64_t> R(K); f(R); return; }
    case r2::Rung::R4: { Ops2<u128> R(K); f(R); return; }
    case r2::Rung::R5:
        switch (c.words) {
        case 3: { Ops2<r2::WordN<3>> R(K); f(R); return; }
        case 4: { Ops2<r2::WordN<4>> R(K); f(R); return; }
        case 5: { Ops2<r2::WordN<5>> R(K); f(R); return; }
        case 6: { Ops2<r2::WordN<6>> R(K); f(R); return; }
        case 8: { Ops2<r2::WordN<8>> R(K); f(R); return; }
        case 10: { Ops2<r2::WordN<10>> R(K); f(R); return; }
        case 12: { Ops2<r2::WordN<12>> R(K); f(R); return; }
        case 16: { Ops2<r2::WordN<16>> R(K); f(R); return; }
        default: break;
        }
        break;
    }
    ENGINE_CHECK_MSG(false, "l23: BIN-3 instantiates the ladder up to R5(N=16), K <= 1024");
}
constexpr uint32_t kDigit3 = 39;   // engine::ladic's digit exponent m(64, 3)
template<class F>
void with_ops3(uint32_t K, F&& f) {
    const uint32_t L = (std::max<uint32_t>(K, 2) + kDigit3 - 1) / kDigit3;
    switch (L) {
    case 1: { Ops3<1> R(K); f(R); return; }
    case 2: { Ops3<2> R(K); f(R); return; }
    case 3: { Ops3<3> R(K); f(R); return; }
    case 4: { Ops3<4> R(K); f(R); return; }
    case 5: { Ops3<5> R(K); f(R); return; }
    case 6: { Ops3<6> R(K); f(R); return; }
    case 7: { Ops3<7> R(K); f(R); return; }
    case 8: { Ops3<8> R(K); f(R); return; }
    default: break;
    }
    ENGINE_CHECK_MSG(false, "l23: BIN-3 instantiates Z/3^K up to 8 digits, K <= 312");
}

} // namespace

// ---------------------------------------------------------------------------
// The lattice cell, adaptively
// ---------------------------------------------------------------------------
LatticeCell read_lattice(uint32_t n, const ReaderOptions& opt) {
    ENGINE_CHECK_MSG(opt.ell == 2 || opt.ell == 3, "l23::read_lattice: ell must be 2 or 3");
    ENGINE_CHECK_MSG(n >= 1, "l23::read_lattice: n >= 1");
    const auto t0 = std::chrono::steady_clock::now();
    LatticeCell out;
    out.ell = opt.ell;
    out.n = n;
    out.d = n + 1;
    out.rank_q_A = rank_q_relation(n);
    const uint32_t need = std::max(opt.m_target, opt.cap);
    uint32_t K = std::max<uint32_t>(opt.k_min ? opt.k_min : (opt.ell == 2 ? 64 : 2 * kDigit3), std::max<uint32_t>(need, 2));
    const uint32_t k_cap = std::min<uint32_t>(opt.k_cap, opt.ell == 2 ? 1024u : 8 * kDigit3);   // the instantiated range
    for (uint32_t run = 0;; ++run) {
        if (opt.ell == 2) {
            const r2::RungChoice c = r2::select_rung(std::min<uint32_t>(K, 4096));
            K = std::min(c.capacity, k_cap);
            with_ops2(K, [&](const auto& R) { lattice_attempt(R, n, opt, opt.cap, out); });
        } else {
            K = std::min(((K + kDigit3 - 1) / kDigit3) * kDigit3, k_cap);
            with_ops3(K, [&](const auto& R) { lattice_attempt(R, n, opt, opt.cap, out); });
        }
        const bool resolved = out.ok && out.m >= need;
        char buf[400];
        std::snprintf(buf, sizeof(buf), "BIN3_ADAPTIVE ell=%u n=%u run=%u rung=%s K=%u e_A=%u e_1=%u m=%u m_target=%u -> %s%s%s",
                      opt.ell, n, run, out.rung.c_str(), K, out.e_A, out.e_1, out.m, need,
                      resolved ? "resolved" : (K >= k_cap ? "UNRESOLVED at k_cap" : "rerun one step up"),
                      out.failures.empty() ? "" : " (", out.failures.empty() ? "" : (out.failures.front() + ")").c_str());
        out.log.emplace_back(buf);
        out.reruns = run;
        if (resolved) break;
        if (K >= k_cap) { out.ok = false; break; }
        K = K + 1;   // the next rung (ell = 2) / the next digit (ell = 3)
    }
    out.seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    return out;
}

// ---------------------------------------------------------------------------
// H^1(Gamma; V / ell^k)
// ---------------------------------------------------------------------------
namespace {
template<class Ops>
ModCell h1_mod_impl(const Ops& R, uint32_t n, uint32_t k, const std::vector<std::pair<uint32_t, int64_t>>& ops, uint32_t mutate) {
    using M = RMat<typename Ops::Elt>;
    ModCell out;
    out.ell = Ops::kEll;
    out.n = n;
    out.k = k;
    const Amalgam<Ops> am(R, n, mutate);
    const M A = am.relation_matrix();
    const M B = am.coboundaries();
    out.h0 = (n + 1) - static_cast<uint32_t>(R.smith(B, 1).size());
    const M Z = rows_as_columns(R, R.kernel_module(A, k), 2 * (n + 1));
    out.length = quotient_length(R, Z, B, k);
    // (slot-23 BIN3O1) T_p Z once per distinct p (the census asks for T_ell and T_ell - 1): the same matrix, computed once.
    const bool reuse = am.rho.fast;
    std::map<uint32_t, M> TZ;
    for (const auto& oc : ops) {
        M X;
        if (reuse) {
            auto it = TZ.find(oc.first);
            if (it == TZ.end()) it = TZ.emplace(oc.first, am.hecke(oc.first, Z)).first;
            else g_hecke_reused.fetch_add(1, std::memory_order_relaxed);
            X = it->second;
        } else {
            X = am.hecke(oc.first, Z);
        }
        if (oc.second) {
            const auto c = R.from_i64(oc.second);
            for (size_t t = 0; t < X.a.size(); ++t) X.a[t] = R.sub(X.a[t], R.mul(c, Z.a[t]));
        }
        out.image_length.push_back(quotient_length(R, X, B, k));
    }
    return out;
}
} // namespace

namespace {
// L23_TERNARY_F3, default on; exactly "0" = the Ops3<1> route; (TER-1 audit F4) anything else warns and stays on.
// One engine::lever::EnvLever (S24 CL-7, F10).
constexpr engine::lever::Policy kTernaryF3Policy{true, engine::lever::Empty::Invalid, engine::lever::Match::Exact01,
                                                 engine::lever::Cache::Once,
                                                 "WARNING l23: L23_TERNARY_F3='%s' is neither 0 nor 1; the F_3 route stays ON "
                                                 "(only exactly 0 turns it off)\n"};
engine::lever::EnvArmRetiredLever g_ternary_f3_reader("L23_TERNARY_F3", kTernaryF3Policy);
} // namespace

bool ternary_f3_reader_enabled() { return g_ternary_f3_reader.enabled(); }
ScopedTernaryF3Reader::ScopedTernaryF3Reader(bool on) : guard_(g_ternary_f3_reader, on) {}
std::string ternary_f3_effective_line() {
    return std::string("EFFECTIVE l23_ternary_f3=") + (ternary_f3_reader_enabled() ? "on" : "off") +
           " source=" + g_ternary_f3_reader.source() + " (h1_mod ell=3 k=1: Ops3F/TritMatrix when on, Ops3<1> when off)";
}

bool bin3o1_enabled() { return bin3o1_on(); }
ScopedBin3o1::ScopedBin3o1(bool on) : guard_(g_bin3o1, on) {}
std::string bin3o1_effective_line() {
    return std::string("EFFECTIVE l23_bin3o1=") + (bin3o1_enabled() ? "on" : "off") + " source=" + g_bin3o1.source() +
           " (amalgam reader: O(d^2) relation matrix, parity-split A, vectorised Taylor shifts; h1_mod ell=2 k=1 on F_2/BitMatrix)";
}
Bin3o1Counters bin3o1_counters() {
    Bin3o1Counters c;
    c.split_accepted = g_split_accepted.load();
    c.split_fallback = g_split_fallback.load();
    c.odd_reduced = g_odd_reduced.load();
    c.rankq_blocks = g_rankq_blocks.load();
    c.f2_cells = g_f2_cells.load();
    c.hecke_reused = g_hecke_reused.load();
    c.rankq_fallback = g_rankq_fallback.load();
    c.parity_fallback = g_parity_fallback.load();
    return c;
}
std::string bin3o1_route_line() {
    const Bin3o1Counters c = bin3o1_counters();
    const auto s = [](uint64_t v) { return std::to_string(v); };
    return "BIN3O1_ROUTE route=" + std::string(bin3o1_enabled() ? "on" : "off") + " split=" + s(c.split_accepted) + "/" +
           s(c.split_accepted + c.split_fallback) + " odd=" + s(c.odd_reduced) + " rankq=" + s(c.rankq_blocks) + " f2=" + s(c.f2_cells) +
           " reused=" + s(c.hecke_reused) + " split_fallback=" + s(c.split_fallback) + " rankq_fallback=" + s(c.rankq_fallback) +
           " parity_fallback=" + s(c.parity_fallback) + " fallback=" + s(c.split_fallback + c.rankq_fallback + c.parity_fallback);
}
uint64_t taylor2_selfcheck(uint32_t n, uint32_t r, uint64_t seed) {
    const Ops2F R(1);
    const Rho<Ops2F> rho(R, n);
    const Ops2<uint8_t> R8(1);
    const Rho<Ops2<uint8_t>> rho8(R8, n);
    const uint32_t d = n + 1;
    RMat<uint8_t> X(d, r, 0);
    uint64_t s = seed;
    for (auto& x : X.a) {
        s = s * 6364136223846793005ULL + 1442695040888963407ULL;
        x = static_cast<uint8_t>((s >> 33) & 1u);
    }
    uint64_t bad = 0;
    RMat<uint8_t> T = X, Ti = X, T8 = X, Ti8 = X;
    rho.T(T);
    rho.Ti(Ti);
    rho8.T(T8);
    rho8.Ti(Ti8);
    for (size_t i = 0; i < X.a.size(); ++i) bad += (T.a[i] != (T8.a[i] & 1u)) + (Ti.a[i] != (Ti8.a[i] & 1u));
    RMat<uint8_t> Z = T;
    rho.Ti(Z);   // T^-1 T = id
    for (size_t i = 0; i < X.a.size(); ++i) bad += Z.a[i] != X.a[i];
    return bad;
}
uint64_t relation_selfcheck(uint32_t ell, uint32_t n, uint32_t k) {
    uint64_t bad = 0;
    auto run = [&](const auto& R) {
        using Ops = std::decay_t<decltype(R)>;
        RMat<typename Ops::Elt> A[2], B[2];
        for (int on = 0; on < 2; ++on) {
            ScopedBin3o1 sc(on != 0);
            const Amalgam<Ops> am(R, n);
            A[on] = am.relation_matrix();
            B[on] = am.coboundaries();
            if (on) {
                typename Amalgam<Ops>::Blocks bl = am.blocks();
                if (!am.parity_structure(bl)) ++bad;
            }
        }
        if (A[0].a.size() != A[1].a.size() || B[0].a.size() != B[1].a.size()) return void(++bad);
        for (size_t t = 0; t < A[0].a.size(); ++t) bad += !R.zero_mod(R.sub(A[0].a[t], A[1].a[t]), R.K);
        for (size_t t = 0; t < B[0].a.size(); ++t) bad += !R.zero_mod(R.sub(B[0].a[t], B[1].a[t]), R.K);
    };
    if (ell == 2) with_ops2(k, run);
    else with_ops3(k, run);
    return bad;
}
uint64_t rows_selfcheck(uint32_t k, uint32_t n, uint32_t r, uint64_t seed) {
    uint64_t bad = 0;
    with_ops2(k, [&](const auto& R) {
        using Ops = std::decay_t<decltype(R)>;
        using E = typename Ops::Elt;
        RMat<E> X(n + 1, r, R.zero()), Y(n + 1, r, R.zero());
        uint64_t s = seed;
        for (size_t i = 0; i < X.a.size(); ++i) {
            s = s * 6364136223846793005ULL + 1442695040888963407ULL;
            X.a[i] = R.from_i64(static_cast<int64_t>(s >> 1));
            s = s * 6364136223846793005ULL + 1442695040888963407ULL;
            Y.a[i] = R.from_i64(static_cast<int64_t>(s >> 1));
        }
        constexpr int NC = 9;
        RMat<E> out[2][NC];
        for (int on = 0; on < 2; ++on) {
            ScopedBin3o1 sc(on != 0);
            const Rho<Ops> rho(R, n);
            out[on][0] = X; rho.T(out[on][0]);
            out[on][1] = X; rho.Ti(out[on][1]);
            out[on][2] = X; rho.U(out[on][2]);
            out[on][3] = mat_add(R, X, Y);
            out[on][4] = mat_sub(R, X, Y);
            out[on][5] = X; rho.word(out[on][5], G2{-7, 4, -2, 1});
            out[on][6] = X; rho.word(out[on][6], G2{1, 9, 0, 1});      // T^9: one shift by 9 on the route
            out[on][7] = X; rho.word(out[on][7], G2{1, -5, 0, 1});     // T^-5
            out[on][8] = X; rho.word(out[on][8], G2{13, 3, 4, 1});     // T^3 S T^4 S ... (runs of 3 and 4)
        }
        for (int c = 0; c < NC; ++c)
            for (size_t t = 0; t < out[0][c].a.size(); ++t) bad += !R.zero_mod(R.sub(out[0][c].a[t], out[1][c].a[t]), R.K);
    });
    return bad;
}

// (TER-1 audit F1) The characteristic-3 Taylor shift against the d^2 / 2 row-operation shift, on the matrices themselves.
uint64_t taylor3_selfcheck(uint32_t n, uint32_t r, uint64_t seed) {
    const Ops3F R(1);
    const Rho<Ops3F> rho(R, n);
    const uint32_t d = n + 1;
    RMat<uint8_t> X(d, r, 0);
    uint64_t s = seed;
    for (auto& x : X.a) {
        s = s * 6364136223846793005ULL + 1442695040888963407ULL;
        x = static_cast<uint8_t>((s >> 33) % 3);
    }
    auto naive = [&](RMat<uint8_t> Y, bool inverse) {   // the generic Rho::T / Ti loop, written out
        for (uint32_t i = 0; i < n; ++i)
            for (uint32_t j = n; j-- > i;) {
                uint8_t* dst = Y.row(j);
                const uint8_t* src = Y.row(j + 1);
                for (uint32_t c = 0; c < r; ++c) dst[c] = inverse ? R.sub(dst[c], src[c]) : R.add(dst[c], src[c]);
            }
        return Y;
    };
    uint64_t bad = 0;
    RMat<uint8_t> T = X, Ti = X;
    rho.T(T);
    rho.Ti(Ti);
    const RMat<uint8_t> Tn = naive(X, false), Tin = naive(X, true);
    for (size_t i = 0; i < X.a.size(); ++i) bad += (T.a[i] != Tn.a[i]) + (Ti.a[i] != Tin.a[i]);
    RMat<uint8_t> Z = T;
    rho.Ti(Z);   // T^-1 T = id
    for (size_t i = 0; i < X.a.size(); ++i) bad += Z.a[i] != X.a[i];
    return bad;
}

ModCell h1_mod(uint32_t ell, uint32_t n, uint32_t k, const std::vector<std::pair<uint32_t, int64_t>>& ops, uint32_t mutate) {
    ENGINE_CHECK_MSG((ell == 2 || ell == 3) && k >= 1, "l23::h1_mod: ell in {2, 3}, k >= 1");
    ModCell out;
    if (ell == 2 && k == 1 && bin3o1_on()) {   // (slot-23 BIN3O1) F_2 on BitMatrix (M4RI)
        g_f2_cells.fetch_add(1, std::memory_order_relaxed);
        const Ops2F R(1);
        out = h1_mod_impl(R, n, k, ops, mutate);
    }
    else if (ell == 2) with_ops2(k, [&](const auto& R) { out = h1_mod_impl(R, n, k, ops, mutate); });
    else if (k == 1 && ternary_f3_reader_enabled()) { const Ops3F R(1); out = h1_mod_impl(R, n, k, ops, mutate); }   // (slot-23 TER-1)
    else with_ops3(k, [&](const auto& R) { out = h1_mod_impl(R, n, k, ops, mutate); });
    return out;
}

// ---------------------------------------------------------------------------
// The layers of H^1(Gamma; Sym^n F_ell^2) (L23-3)
// ---------------------------------------------------------------------------
namespace {
// The columns of X that extend a basis of span(Y) to one of span(Y) + span(X), over F_ell (column rank profile of [Y | X]).
template<class Ops>
RMat<typename Ops::Elt> complement_cols(const Ops& R, const RMat<typename Ops::Elt>& X, const RMat<typename Ops::Elt>& Y) {
    using E = typename Ops::Elt;
    ENGINE_CHECK(X.rows == Y.rows || Y.cols == 0);
    if (X.cols == 0) return X;
    engine::DenseBuf<uint64_t> b(X.rows, Y.cols + X.cols);
    for (uint32_t i = 0; i < X.rows; ++i) {
        for (uint32_t j = 0; j < Y.cols; ++j) b.at(i, j) = R.lowval(Y.at(i, j), 1);
        for (uint32_t j = 0; j < X.cols; ++j) b.at(i, Y.cols + j) = R.lowval(X.at(i, j), 1);
    }
    const engine::ModCtx<uint64_t> F(Ops::kEll);
    const engine::elim::RrefResult rr = engine::elim::column_rank_profile(F, engine::DenseView<uint64_t>(b));
    std::vector<uint32_t> keep;
    for (uint32_t c : rr.pivots) if (c >= Y.cols) keep.push_back(c - Y.cols);
    RMat<E> out(X.rows, static_cast<uint32_t>(keep.size()), R.zero());
    for (uint32_t i = 0; i < X.rows; ++i)
        for (uint32_t j = 0; j < keep.size(); ++j) out.at(i, j) = R.reduce(X.at(i, keep[j]), 1);
    return out;
}
// X * (the first X.cols entries of the kernel rows of [Phi | W]): the columns x of span X whose image under the map with matrix
// Phi (columns parallel to X) lies in span W, over F_ell.
template<class Ops>
RMat<typename Ops::Elt> preimage_cols(const Ops& R, const RMat<typename Ops::Elt>& X, const RMat<typename Ops::Elt>& Phi,
                                      const RMat<typename Ops::Elt>& W) {
    using E = typename Ops::Elt;
    ENGINE_CHECK(Phi.cols == X.cols);
    if (X.cols == 0) return X;
    const RMat<E> PW = W.cols ? hcat(R, Phi, W) : Phi;
    const HowellOut<E> K = R.kernel_module(PW, 1);
    std::vector<uint32_t> rows;
    for (uint32_t i = 0; i < K.H.rows; ++i) {
        bool nz = false;
        for (uint32_t j = 0; j < X.cols && !nz; ++j) nz = !R.zero_mod(K.H.at(i, j), 1);
        if (nz) rows.push_back(i);
    }
    RMat<E> Ct(X.cols, static_cast<uint32_t>(rows.size()), R.zero());
    for (uint32_t c = 0; c < rows.size(); ++c)
        for (uint32_t j = 0; j < X.cols; ++j) Ct.at(j, c) = R.reduce(K.H.at(rows[c], j), 1);
    RMat<E> Z = mat_mul(R, X, Ct);
    for (auto& x : Z.a) x = R.reduce(x, 1);
    return Z;
}
template<class Ops>
Jordan01 jordan_on(const Ops& R, const Amalgam<Ops>& am, const RMat<typename Ops::Elt>& X, const RMat<typename Ops::Elt>& Y) {
    using M = RMat<typename Ops::Elt>;
    const uint32_t ell = Ops::kEll;
    Jordan01 J;
    J.dim = quotient_length(R, X, Y, 1);
    if (J.dim == 0) return J;
    auto red1 = [&](M P) { for (auto& x : P.a) x = R.reduce(x, 1); return P; };
    const M TX = red1(am.hecke(ell, X));
    J.rank = quotient_length(R, TX, Y, 1);
    auto invalid = [&J]() {   // T_ell does not preserve the subquotient: no Jordan type (never on a correct operator)
        J.valid = false;
        J.other = J.dim;
        J.eig0.clear();
        J.eig1.clear();
        return J;
    };
    uint32_t gen[2] = {0, 0};
    for (uint32_t lam = 0; lam <= 1; ++lam) {
        // r_j = dim of (T - lam)^j X mod Y; P <- (T - lam) P, one T_ell per step (L23_3_audit.md m3: no binomials)
        std::vector<uint32_t> r = {J.dim};
        M P = lam ? red1(mat_sub(R, M(TX), X)) : TX;
        for (uint32_t j = 1; j <= J.dim; ++j) {
            if (j > 1) {
                M TP = am.hecke(ell, P);
                P = red1(lam ? mat_sub(R, std::move(TP), P) : std::move(TP));
            }
            const uint32_t rj = quotient_length(R, P, Y, 1);
            if (rj > r.back()) return invalid();
            if (rj == r.back()) break;
            r.push_back(rj);
            if (rj == 0) break;
        }
        std::vector<uint32_t> blocks;
        if (!jordan_blocks_from_ranks(r, blocks)) return invalid();
        gen[lam] = J.dim - r.back();
        (lam == 0 ? J.eig0 : J.eig1) = blocks;
    }
    J.other = J.dim - gen[0] - gen[1];
    return J;
}

template<class Ops>
LayerCell h1_layers_impl(const Ops& R, uint32_t n, uint32_t a_max, uint32_t E, uint32_t mutate) {
    using M = RMat<typename Ops::Elt>;
    const uint32_t ell = Ops::kEll, d = n + 1;
    LayerCell out;
    out.ell = ell;
    out.n = n;
    out.a_max = a_max;
    out.E = E;
    const Amalgam<Ops> am(R, n, mutate == 1 ? 1 : 0);
    const M A = am.relation_matrix();
    const M B = am.coboundaries();
    auto red1 = [&](M X) { for (auto& x : X.a) x = R.reduce(x, 1); return X; };
    const M B1 = red1(B);
    out.h0 = d - static_cast<uint32_t>(R.smith(B, 1).size());
    const M Z1 = red1(rows_as_columns(R, R.kernel_module(A, 1), 2 * d));
    out.h1 = quotient_length(R, Z1, B1, 1);
    const M X1 = complement_cols(R, Z1, B1);   // a basis of H^1 (mod B^1)
    ENGINE_CHECK_MSG(X1.cols == out.h1, "l23::h1_layers: complement basis");
    // (B): (c(S), c(U)) modulo im(S - 1) + im(U - 1)
    M Bdiag(2 * d, 2 * d, R.zero());
    for (uint32_t i = 0; i < d; ++i)
        for (uint32_t j = 0; j < d; ++j) {
            Bdiag.at(i, j) = B1.at(i, j);
            Bdiag.at(d + i, d + j) = B1.at(d + i, j);
        }
    const M Zbr = complement_cols(R, preimage_cols(R, X1, X1, Bdiag), B1);
    out.dim_br = Zbr.cols;
    // (!): Phi(c) = (c(T), c(-I)) against W = ((T - 1) v, (-I - 1) v)
    auto phi = [&](const M& C) {
        typename Amalgam<Ops>::Vals v = am.values(C);
        M cT = v.cT;
        if (mutate == 3) { cT = v.cSinv; }   // NEGATIVE CONTROL: c(T) without S^-1 c(U)
        M s = v.cS;
        am.rho.S(s);
        M cmI = mat_add(R, v.cS, s);
        return red1(am.stack(cT, cmI));
    };
    M W(2 * d, d, R.zero());
    {
        M t = am.rho.identity();
        am.rho.T(t);
        const M I = am.rho.identity();
        t = mat_sub(R, std::move(t), I);
        for (uint32_t i = 0; i < d; ++i)
            for (uint32_t j = 0; j < d; ++j) {
                W.at(i, j) = R.reduce(t.at(i, j), 1);
                if (n % 2 == 1 && i == j) W.at(d + i, j) = R.reduce(R.from_i64(-2), 1);   // (-I - 1) = ((-1)^n - 1) I
            }
    }
    const M Zcusp = complement_cols(R, preimage_cols(R, X1, phi(X1), W), B1);
    out.dim_cusp = Zcusp.cols;
    auto cusp_cap = [&](const M& Y) {   // dim(span Y mod B cap H^1_!) for a basis Y mod B
        return Y.cols ? Y.cols - quotient_length(R, phi(Y), W, 1) : 0u;
    };
    // (U): H^1(V)/ell and T^1/ell
    const M Zsub = complement_cols(R, red1(rows_as_columns(R, R.kernel_module(A, E), 2 * d)), B1);
    out.dim_sub = Zsub.cols;
    M T1c(2 * d, 0, R.zero());
    if (a_max > 0) {
        const HowellOut<typename Ops::Elt> Uk = R.kernel_module(B, a_max);   // u in V / ell^a with delta u = 0
        M Ut(d, Uk.H.rows, R.zero());
        for (uint32_t c = 0; c < Uk.H.rows; ++c)
            for (uint32_t j = 0; j < d; ++j) Ut.at(j, c) = Uk.H.at(c, j);
        const M Du = mat_mul(R, B, Ut);
        uint64_t la = 1;
        for (uint32_t i = 0; i < a_max; ++i) la *= ell;
        M Tc(2 * d, Du.cols, R.zero());
        for (size_t t = 0; t < Du.a.size(); ++t) {
            const uint64_t x = R.lowval(Du.a[t], a_max + 1);
            ENGINE_CHECK_MSG(x % la == 0, "l23::h1_layers: delta u not divisible by ell^a");
            Tc.a[t] = R.from_i64(static_cast<int64_t>(x / la));
        }
        out.t1_cocycles = am.cocycles(Tc, 1);
        T1c = complement_cols(R, Tc, B1);
    } else {
        out.t1_cocycles = true;
    }
    out.dim_t1 = T1c.cols;
    out.dim_cusp_br = cusp_cap(Zbr);
    out.dim_cusp_sub = cusp_cap(Zsub);
    out.dim_cusp_t1 = cusp_cap(T1c);
    out.dim_t1_br = out.dim_t1 + out.dim_br - (T1c.cols + Zbr.cols ? quotient_length(R, hcat(R, T1c, Zbr), B1, 1) : 0u);
    // stability and the Jordan types of T_ell
    auto stable = [&](const M& Y) {
        if (Y.cols == 0) return true;
        return quotient_length(R, red1(am.hecke(ell, Y)), hcat(R, Y, B1), 1) == 0;
    };
    out.br_stable = stable(Zbr);
    out.cusp_stable = stable(Zcusp);
    out.j_whole = jordan_on(R, am, X1, B1);
    out.j_br = jordan_on(R, am, Zbr, B1);
    out.j_quot = jordan_on(R, am, X1, Zbr.cols ? hcat(R, Zbr, B1) : B1);
    out.j_cusp = jordan_on(R, am, Zcusp, B1);
    // (L) at ell = 3
    if (ell == 3 && n % 2 == 0) {
        // (L23_3_audit.md n2) the four are redundant: every parabolic of Gamma(3) is Gamma-conjugate to T^{+-3} and
        // c(h g h^-1) = h c(g) mod 3 for g in Gamma(3) (trivial action), so L = {c : c(T^3) = 0}; the other three are a self-check.
        const G2 par[4] = {G2{1, 3, 0, 1}, G2{1, 0, -3, 1}, G2{-2, 3, -3, 4}, G2{4, 3, -3, -2}};   // the cusps oo, 0, 1, -1
        for (const G2& g : par) ENGINE_CHECK_MSG(g.a * g.d - g.b * g.c == 1 && pmod(g.b, 3) == 0 && pmod(g.c, 3) == 0 &&
                                                     pmod(g.a, 3) == 1 && pmod(g.d, 3) == 1, "l23::h1_layers: not in Gamma(3)");
        auto ev = [&](const M& C) {
            const typename Amalgam<Ops>::Vals v = am.values(C);
            M S4(4 * d, C.cols, R.zero());
            for (uint32_t g = 0; g < 4; ++g) {
                const M x = am.ev(v, par[g]);
                std::copy(x.a.begin(), x.a.end(), S4.a.begin() + static_cast<std::ptrdiff_t>(static_cast<size_t>(g) * d * C.cols));
            }
            return red1(S4);
        };
        const M ZL = complement_cols(R, preimage_cols(R, X1, ev(X1), M(4 * d, 0, R.zero())), B1);
        out.dim_L = ZL.cols;
        out.l_done = true;
        if (ZL.cols) {
            out.L_in_T1 = quotient_length(R, ZL, hcat(R, T1c, B1), 1) == 0;
            out.L_in_sub = quotient_length(R, ZL, hcat(R, Zsub, B1), 1) == 0;
            out.L_stable = stable(ZL);
            out.dim_L_br = out.dim_L + out.dim_br - quotient_length(R, hcat(R, ZL, Zbr), B1, 1);
        } else {
            out.L_stable = true;
        }
    }
    return out;
}
} // namespace

bool jordan_blocks_from_ranks(const std::vector<uint32_t>& r, std::vector<uint32_t>& blocks) {
    // r_j = rank (T - lam)^j, j = 0 .. m, strictly decreasing; b_j = r_{j-1} - r_j = #{blocks of size >= j} must be
    // non-increasing (a theorem for an operator; violated when T does not preserve the subquotient, L23_3_audit.md M1)
    blocks.clear();
    for (size_t j = 1; j < r.size(); ++j)
        if (r[j] >= r[j - 1]) return false;
    for (size_t j = 1; j < r.size(); ++j) {
        const uint32_t bj = r[j - 1] - r[j];
        const uint32_t bj1 = (j + 1 < r.size()) ? r[j] - r[j + 1] : 0;
        if (bj < bj1) { blocks.clear(); return false; }
        for (uint32_t t = 0; t < bj - bj1; ++t) blocks.push_back(static_cast<uint32_t>(j));
    }
    std::sort(blocks.rbegin(), blocks.rend());
    ENGINE_CHECK_MSG(r.empty() || blocks.size() <= r.front(), "l23::jordan_blocks_from_ranks: more blocks than the dimension");
    return true;
}

LayerCell h1_layers(uint32_t ell, uint32_t n, uint32_t a_max, uint32_t E, uint32_t mutate) {
    ENGINE_CHECK_MSG(ell == 2 || ell == 3, "l23::h1_layers: ell in {2, 3}");
    ENGINE_CHECK_MSG(E >= 1 && a_max <= 30, "l23::h1_layers: E >= 1, a_max <= 30");
    const auto t0 = std::chrono::steady_clock::now();
    const uint32_t K = std::max<uint32_t>({E, a_max + 1, 2});
    LayerCell out;
    if (ell == 2) with_ops2(K, [&](const auto& R) { out = h1_layers_impl(R, n, a_max, E, mutate); });
    else with_ops3(K, [&](const auto& R) { out = h1_layers_impl(R, n, a_max, E, mutate); });
    out.seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    return out;
}

// ---------------------------------------------------------------------------
// oddn_l2.py's columns
// ---------------------------------------------------------------------------
OddnRow oddn_row(uint32_t n, uint32_t K, uint32_t mutate) {
    ENGINE_CHECK_MSG(K >= 2 && K <= 8, "l23::oddn_row: 2 <= K <= 8");
    using R8 = Ops2<uint8_t>;
    using M = RMat<uint8_t>;
    const R8 R(8);
    const Amalgam<R8> am(R, n, mutate);
    const M A = am.relation_matrix();
    const M B = am.coboundaries();
    const uint32_t d = n + 1;
    auto Z = [&](uint32_t k) { return rows_as_columns(R, R.kernel_module(A, k), 2 * d); };
    auto qd = [&](const M& X, const M& Bk, uint32_t k) { return quotient_length(R, X, Bk, k); };
    const M Z1 = Z(1), Z2 = Z(2), ZK = Z(K);
    M Z2red = Z2;
    for (auto& x : Z2red.a) x = R.reduce(x, 1);
    OddnRow row;
    row.n = n;
    auto& v = row.v;
    const uint32_t h0 = d - static_cast<uint32_t>(R.smith(B, 1).size());
    uint32_t h0d = 0;
    for (uint32_t a = 0; a <= n / 3; ++a) if ((n - 3 * a) % 2 == 0) ++h0d;
    const uint32_t h1 = qd(Z1, B, 1);
    const uint32_t d1 = qd(Z2red, B, 1);
    const uint32_t dimK = qd(ZK, B, K);
    M twoZ = ZK;
    for (auto& x : twoZ.a) x = R.add(x, x);
    const uint32_t exp2 = qd(twoZ, B, K) == 0 ? 1 : 0;
    v = {h0, h0d, h1, n / 2 + 1, d1, h1 - d1, exp2, dimK};
    for (uint32_t p : {3u, 5u, 7u}) v.push_back(qd(am.hecke(p, ZK), B, K));
    for (uint32_t p : {3u, 5u, 7u}) v.push_back(qd(am.hecke(p, Z1), B, 1));
    const M T2Z = am.hecke(2, Z1);
    const M T2T2Z = am.hecke(2, T2Z);
    v.push_back(qd(T2Z, B, 1));
    v.push_back(qd(T2T2Z, B, 1));
    v.push_back(qd(mat_sub(R, T2Z, Z1), B, 1));
    v.push_back(qd(mat_sub(R, T2T2Z, T2Z), B, 1) == 0 ? 1 : 0);
    const M T2sub = am.hecke(2, Z2red);
    v.push_back(qd(T2sub, B, 1));
    const M subB = hcat(R, Z2red, B);
    v.push_back(qd(T2sub, subB, 1) == 0 ? 1 : 0);
    v.push_back(qd(T2Z, subB, 1) == 0 ? 1 : 0);
    v.push_back(qd(am.hecke(2, ZK), B, K));
    return row;
}

// ---------------------------------------------------------------------------
// The structural rho against the dense Sym^n matrix of [C] def:rhon
// ---------------------------------------------------------------------------
namespace {
template<class Ops>
RMat<typename Ops::Elt> dense_sym(const Ops& R, const G2& g, uint32_t n) {
    using E = typename Ops::Elt;
    const uint32_t d = n + 1;
    std::vector<std::vector<E>> C(d);
    for (uint32_t m = 0; m <= n; ++m) {
        C[m].assign(m + 1, R.one());
        for (uint32_t i = 1; i < m; ++i) C[m][i] = R.add(C[m - 1][i - 1], C[m - 1][i]);
    }
    auto pows = [&](int64_t x) {
        std::vector<E> p(d, R.one());
        for (uint32_t i = 1; i < d; ++i) p[i] = R.mul(p[i - 1], R.from_i64(x));
        return p;
    };
    const std::vector<E> pa = pows(g.a), pb = pows(g.b), pc = pows(g.c), pd = pows(g.d);
    RMat<E> M(d, d, R.zero());
    for (uint32_t j = 0; j <= n; ++j)
        for (uint32_t i1 = 0; i1 <= n - j; ++i1) {
            const E c1 = R.mul(R.mul(C[n - j][i1], pa[n - j - i1]), pc[i1]);
            for (uint32_t i2 = 0; i2 <= j; ++i2) {
                const E c2 = R.mul(R.mul(C[j][i2], pb[j - i2]), pd[i2]);
                M.at(i1 + i2, j) = R.add(M.at(i1 + i2, j), R.mul(c1, c2));
            }
        }
    return M;
}
template<class Ops>
uint32_t selfcheck_impl(const Ops& R, uint32_t n, uint32_t k) {
    using M = RMat<typename Ops::Elt>;
    const Rho<Ops> rho(R, n);
    uint32_t bad = 0;
    auto cmp = [&](const M& X, const M& Y) {
        for (size_t t = 0; t < X.a.size(); ++t) if (!R.zero_mod(R.sub(X.a[t], Y.a[t]), k)) ++bad;
    };
    struct Case { G2 g; int kind; uint32_t p; };   // kind 0: word / letter; 1: eps; 2: diag(1,p); 3: r^-1 diag(1,p)
    std::vector<Case> cases = {{kS, 10, 0}, {kSinv, 11, 0}, {kT, 12, 0}, {kTinv, 13, 0}, {kU, 14, 0},
                               {G2{-1, 0, 0, 1}, 1, 0}, {G2{1, 0, 0, 5}, 2, 5},
                               {G2{2, 3, 5, 8}, 0, 0}, {G2{-7, 4, -2, 1}, 0, 0}, {G2{1, 0, -3, 1}, 0, 0}, {G2{0, 1, -1, 1}, 0, 0}};
    for (uint32_t p : {2u, 3u, 7u})
        for (const G2& r : coset_reps(p)) cases.push_back({gmul(ginv(r), G2{1, 0, 0, static_cast<int64_t>(p)}), 3, p});
    std::vector<G2> repsr;
    for (const Case& c : cases) {
        M X = rho.identity();
        switch (c.kind) {
        case 10: rho.S(X); break;
        case 11: rho.Sinv(X); break;
        case 12: rho.T(X); break;
        case 13: rho.Ti(X); break;
        case 14: rho.U(X); break;
        case 1: rho.eps(X); break;
        case 2: rho.D(X, c.p); break;
        case 3: {
            rho.D(X, c.p);
            // r^-1 = (r^-1 g) diag(1, p)^-1: recover r^-1 in SL_2(Z) from the case matrix
            const G2 rinv{c.g.a, c.g.b / static_cast<int64_t>(c.p), c.g.c, c.g.d / static_cast<int64_t>(c.p)};
            rho.word(X, rinv);
            break;
        }
        default: rho.word(X, c.g); break;
        }
        cmp(X, dense_sym(R, c.g, n));
    }
    return bad;
}
} // namespace

uint32_t rho_selfcheck(uint32_t ell, uint32_t n, uint32_t k) {
    uint32_t bad = 0;
    if (ell == 2) with_ops2(k, [&](const auto& R) { bad = selfcheck_impl(R, n, k); });
    else with_ops3(k, [&](const auto& R) { bad = selfcheck_impl(R, n, k); });
    return bad;
}

} // namespace l23
