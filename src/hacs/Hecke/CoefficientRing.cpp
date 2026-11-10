// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// Hecke/CoefficientRing.cpp -- hecke-extensions slot 8, WP-3 (E8 item 3).
// See CoefficientRing.h and [E] v1.9.0-pre.1 sec:ring.

#include "CoefficientRing.h"

#include "../LocalSystemSL2Z.h"
#include "../../ModArithmetic/EngineCheck.h"

#include <limits>
#include <type_traits>
#include <utility>

namespace hecke {

uint64_t ExactKthRoot(uint64_t m, uint32_t k) {
    // (slot-11 WP-12) k == 1 is refused like k == 0: the bisection bound below
    // shifted by 64/k + 1 = 65 at k = 1 (undefined; found by UBSan in the
    // Check build).  No caller passes k < 2 (the ring has k >= 2), and on x86
    // the old shift wrapped to hi = 2, so every k = 1 call already returned 0.
    if (k < 2 || m < 4) return 0;
    // Largest r with r^k <= m, by bisection on [1, 2^(64/k) + 1].
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
    // ell must be PRIME (Z/ell^k is a chain ring only then): 625 = 25^2 is
    // not Z/ell^2.  (slot-11 WP-12, F-2) Deterministic Miller-Rabin
    // (IsPrimeU64), O(log ell) -- the slot-8 trial division up to sqrt(ell)
    // (ENGINE_API.md Sec. 13.12.3's blocker) is kept only as the validation's
    // oracle.  The preferred route does not test at all: a declared ring
    // (LocalSystemSimplicialComplexT::declare_coefficient_ring) carries
    // (ell, k) and EquivariantCohomologyOnK reads it instead of calling this.
    if (!IsPrimeU64(lo)) return 0;
    return lo;
}

namespace {
uint64_t MulModU64(uint64_t a, uint64_t b, uint64_t m) {
    return static_cast<uint64_t>((static_cast<unsigned __int128>(a) * b) % m);
}
uint64_t PowModU64(uint64_t a, uint64_t e, uint64_t m) {
    uint64_t r = 1 % m;
    a %= m;
    while (e > 0) {
        if (e & 1) r = MulModU64(r, a, m);
        a = MulModU64(a, a, m);
        e >>= 1;
    }
    return r;
}
}  // namespace

// Deterministic for every n < 2^64: the first twelve primes as bases are a
// witness set for n < 3.317 * 10^24 (Sorenson-Webster 2015; Jaeschke 1993
// for the first seven below 3.4 * 10^14).  Small n by the base list itself.
bool IsPrimeU64(uint64_t n) {
    static constexpr uint64_t kBases[12] = {2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37};
    if (n < 2) return false;
    for (uint64_t b : kBases)
        if (n % b == 0) return n == b;
    uint64_t d = n - 1;
    uint32_t s = 0;
    while ((d & 1) == 0) { d >>= 1; ++s; }
    for (uint64_t a : kBases) {
        uint64_t x = PowModU64(a, d, n);
        if (x == 1 || x == n - 1) continue;
        bool composite = true;
        for (uint32_t r = 1; r < s; ++r) {
            x = MulModU64(x, x, n);
            if (x == n - 1) { composite = false; break; }
        }
        if (composite) return false;
    }
    return true;
}

template<class Elt>
engine::ModCtx<Elt> CoefficientRingContext(const std::string& who, uint64_t ell, uint32_t k,
                                           SmallEllPolicy small_ell) {
    ENGINE_CHECK_MSG(k >= 2, "CoefficientRingContext: k >= 2 (F_ell is ModCtx(ell))");
    // (slot-9 WP-3) Lifted for the Bredon policy only (EquivariantCohomologyOnK).
    if (IsBredonPrime(ell) && small_ell != SmallEllPolicy::Bredon) {
        RefuseCoefficientRing(who, k, " at ell = " + std::to_string(ell) +
                                          " (6 is not a unit of Z/ell^k: the model complex is not C (x) Z/ell^k; "
                                          "[E] sec:ring (b3), line 2)");
    }
    unsigned __int128 acc = 1;
    for (uint32_t i = 0; i < k; ++i) {
        acc *= ell;
        if (acc >= (static_cast<unsigned __int128>(1) << 64)) {
            RefuseCoefficientRing(who, k, " at ell = " + std::to_string(ell) +
                                              " (ell^k >= 2^64, outside ENGINE_API.md Sec. 13's scope)");
        }
    }
    if (acc > static_cast<unsigned __int128>(std::numeric_limits<Elt>::max())) {
        RefuseCoefficientRing(who, k, " at ell = " + std::to_string(ell) +
                                          " in the uint32_t instantiation (ell^k >= 2^32; the uint64_t one honours it)");
    }
    return engine::ModCtx<Elt>::prime_power(static_cast<Elt>(ell), k);
}

template<class Elt>
FlatMatrixT<Elt> RhoOfExactRing(const engine::ModCtx<Elt>& ctx, const Mat22& g, uint32_t n_sym) {
    ENGINE_CHECK_MSG(!ctx.is_field(), "RhoOfExactRing: composite-only (Z/ell^k)");
    std::array<Elt, 4> r;
    if constexpr (std::is_same_v<Elt, uint32_t>) {
        const std::array<uint32_t, 4> a = g.reduced_mod(ctx.p);
        r = {a[0], a[1], a[2], a[3]};
    } else {
        const std::array<uint64_t, 4> a = g.reduced_mod64(ctx.p);
        r = {a[0], a[1], a[2], a[3]};
    }
    return ComputeSymmetricPowerMatrixRing<Elt>(ctx, r[0], r[1], r[2], r[3], n_sym);
}

template<class Elt>
std::vector<typename LocalSystemSimplicialComplexT<Elt>::SimplexRelation>
GenerateSL2ZRelationsRing(const SimplicialComplex& K, uint32_t n_sym, const engine::ModCtx<Elt>& ctx) {
    using Rel = typename LocalSystemSimplicialComplexT<Elt>::SimplexRelation;
    ENGINE_CHECK_MSG(!ctx.is_field(), "GenerateSL2ZRelationsRing: composite-only (Z/ell^k)");
    // The row structure (indices, tags, order) is the field generator's, taken
    // verbatim; only the matrices are recomputed over Z/ell^k.
    const std::vector<Rel> field = GenerateSL2ZRelationsElt<Elt>(K, n_sym, ctx.ell);
    const auto& seeds = SL2ZRelationSeeds_7VertexModel();
    ENGINE_CHECK_MSG(field.size() >= seeds.size(), "GenerateSL2ZRelationsRing: fewer relations than seeds");
    std::vector<std::pair<Mat22, FlatMatrixT<Elt>>> cache;
    auto rho = [&](const Mat22& g) -> const FlatMatrixT<Elt>& {
        for (const auto& e : cache) if (e.first == g) return e.second;
        cache.emplace_back(g, RhoOfExactRing<Elt>(ctx, g, n_sym));
        return cache.back().second;
    };
    const Mat22 neg_I = Mat22::I().neg();
    std::vector<Rel> out;
    out.reserve(field.size());
    for (size_t r = 0; r < field.size(); ++r) {
        const Rel& f = field[r];
        const bool seed = r < seeds.size();
        if (seed) ENGINE_CHECK_MSG(f.tag == seeds[r].tag, "GenerateSL2ZRelationsRing: seed order mismatch");
        else ENGINE_CHECK_MSG(f.tag.compare(0, 4, "negI") == 0, "GenerateSL2ZRelationsRing: expected a FIX-negI row");
        out.push_back(Rel(f.q_dim, f.s_lhs_idx_in_Kq, f.s_rhs_idx_in_Kq, rho(seed ? seeds[r].g_r : neg_I), f.tag));
    }
    return out;
}

// (slot-13 WP-5) See the header.
template<class Elt>
FlatMatrixT<Elt> RhoHidaOfExactRing(const engine::ModCtx<Elt>& ctx, const Mat22& g, uint32_t n_sym) {
    Mat22 t = g;
    std::swap(t.b, t.c);   // g^T (exact)
    return RhoOfExactRing<Elt>(ctx, t, n_sym).transpose();
}

template<class Elt>
std::vector<typename LocalSystemSimplicialComplexT<Elt>::SimplexRelation>
GenerateSL2ZRelationsRingHida(const SimplicialComplex& K, uint32_t n_sym, const engine::ModCtx<Elt>& ctx) {
    using Rel = typename LocalSystemSimplicialComplexT<Elt>::SimplexRelation;
    ENGINE_CHECK_MSG(!ctx.is_field(), "GenerateSL2ZRelationsRingHida: composite-only (Z/ell^k)");
    const std::vector<Rel> sym = GenerateSL2ZRelationsRing<Elt>(K, n_sym, ctx);
    const auto& seeds = SL2ZRelationSeeds_7VertexModel();
    const FlatMatrixT<Elt> neg_I = RhoHidaOfExactRing<Elt>(ctx, Mat22::I().neg(), n_sym);
    std::vector<Rel> out;
    out.reserve(sym.size());
    for (size_t r = 0; r < sym.size(); ++r) {
        const Rel& f = sym[r];
        const bool seed = r < seeds.size();
        out.push_back(Rel(f.q_dim, f.s_lhs_idx_in_Kq, f.s_rhs_idx_in_Kq,
                          seed ? RhoHidaOfExactRing<Elt>(ctx, seeds[r].g_r, n_sym) : neg_I, f.tag));
    }
    return out;
}

// (slot-9 WP-4) See the header.  The row structure is the field generator's,
// verbatim; only the matrices are recomputed over Z/ell^k.
template<class Elt>
std::vector<typename LocalSystemSimplicialComplexT<Elt>::SimplexRelation>
GenerateSL2ZBoundaryRelationsRing(const SimplicialComplex& K_bd, uint32_t n_sym, const engine::ModCtx<Elt>& ctx) {
    using Rel = typename LocalSystemSimplicialComplexT<Elt>::SimplexRelation;
    ENGINE_CHECK_MSG(!ctx.is_field(), "GenerateSL2ZBoundaryRelationsRing: composite-only (Z/ell^k)");
    const std::vector<Rel> field = GenerateSL2ZBoundaryRelationsElt<Elt>(K_bd, n_sym, ctx.ell);
    ENGINE_CHECK_MSG(!field.empty() && field[0].tag == "cusp_T", "GenerateSL2ZBoundaryRelationsRing: expected the cusp row first");
    const FlatMatrixT<Elt> rho_T = RhoOfExactRing<Elt>(ctx, Mat22::T(), n_sym);
    const FlatMatrixT<Elt> rho_neg_I = RhoOfExactRing<Elt>(ctx, Mat22::I().neg(), n_sym);
    std::vector<Rel> out;
    out.reserve(field.size());
    for (size_t r = 0; r < field.size(); ++r) {
        const Rel& f = field[r];
        if (r > 0) ENGINE_CHECK_MSG(f.tag.compare(0, 7, "negI_bd") == 0, "GenerateSL2ZBoundaryRelationsRing: expected a FIX-negI row");
        out.push_back(Rel(f.q_dim, f.s_lhs_idx_in_Kq, f.s_rhs_idx_in_Kq, r == 0 ? rho_T : rho_neg_I, f.tag));
    }
    return out;
}

template engine::ModCtx<uint32_t> CoefficientRingContext<uint32_t>(const std::string&, uint64_t, uint32_t,
                                                                   SmallEllPolicy);
template engine::ModCtx<uint64_t> CoefficientRingContext<uint64_t>(const std::string&, uint64_t, uint32_t,
                                                                   SmallEllPolicy);
template FlatMatrixT<uint32_t> RhoOfExactRing<uint32_t>(const engine::ModCtx<uint32_t>&, const Mat22&, uint32_t);
template FlatMatrixT<uint64_t> RhoOfExactRing<uint64_t>(const engine::ModCtx<uint64_t>&, const Mat22&, uint32_t);
template std::vector<typename LocalSystemSimplicialComplexT<uint32_t>::SimplexRelation>
GenerateSL2ZRelationsRing<uint32_t>(const SimplicialComplex&, uint32_t, const engine::ModCtx<uint32_t>&);
template std::vector<typename LocalSystemSimplicialComplexT<uint64_t>::SimplexRelation>
GenerateSL2ZRelationsRing<uint64_t>(const SimplicialComplex&, uint32_t, const engine::ModCtx<uint64_t>&);
template std::vector<typename LocalSystemSimplicialComplexT<uint32_t>::SimplexRelation>
GenerateSL2ZBoundaryRelationsRing<uint32_t>(const SimplicialComplex&, uint32_t, const engine::ModCtx<uint32_t>&);
template std::vector<typename LocalSystemSimplicialComplexT<uint64_t>::SimplexRelation>
GenerateSL2ZBoundaryRelationsRing<uint64_t>(const SimplicialComplex&, uint32_t, const engine::ModCtx<uint64_t>&);

template FlatMatrixT<uint32_t> RhoHidaOfExactRing<uint32_t>(const engine::ModCtx<uint32_t>&, const Mat22&, uint32_t);
template FlatMatrixT<uint64_t> RhoHidaOfExactRing<uint64_t>(const engine::ModCtx<uint64_t>&, const Mat22&, uint32_t);
template std::vector<typename LocalSystemSimplicialComplexT<uint32_t>::SimplexRelation>
GenerateSL2ZRelationsRingHida<uint32_t>(const SimplicialComplex&, uint32_t, const engine::ModCtx<uint32_t>&);
template std::vector<typename LocalSystemSimplicialComplexT<uint64_t>::SimplexRelation>
GenerateSL2ZRelationsRingHida<uint64_t>(const SimplicialComplex&, uint32_t, const engine::ModCtx<uint64_t>&);

} // namespace hecke
