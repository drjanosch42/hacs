// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// (hecke-extensions slot 15, WP-7) The Heilbronn--Merel action H_{N,p} on the
// cokernel presentation -- see HeckeLevelNHeilbronn.h ([C] sec:levelN,
// lem:merelpath, prop:heilbronn, rmk:heilbronnconv).
#include "HeckeLevelNHeilbronn.h"

#include <algorithm>

#include "CoefficientRing.h"   // hecke::IsPrimeU64
#include "../../ModArithmetic/EngineCheck.h"

namespace hecke {
namespace level {

namespace {

using u128 = unsigned __int128;

uint64_t InvModP(uint64_t a, uint64_t p) {
    // p prime, a != 0 mod p: a^{p-2}.
    uint64_t r = 1, b = a % p, e = p - 2;
    while (e) {
        if (e & 1) r = static_cast<uint64_t>(static_cast<u128>(r) * b % p);
        b = static_cast<uint64_t>(static_cast<u128>(b) * b % p);
        e >>= 1;
    }
    return r;
}

int64_t ModN(int64_t x, uint64_t N) {
    const int64_t r = x % static_cast<int64_t>(N);
    return r < 0 ? r + static_cast<int64_t>(N) : r;
}

thread_local bool t_force_wide = false;

} // namespace

ScopedForceWideHeilbronn::ScopedForceWideHeilbronn() : prev_(t_force_wide) { t_force_wide = true; }
ScopedForceWideHeilbronn::~ScopedForceWideHeilbronn() { t_force_wide = prev_; }
bool ScopedForceWideHeilbronn::active() { return t_force_wide; }

// ---- lem:merelpath -------------------------------------------------------------------
std::vector<Mat22> MerelSet(uint64_t n) {
    ENGINE_CHECK_MSG(n >= 1 && n < (1ull << 20), "MerelSet: n in [1, 2^20)");
    std::vector<Mat22> out;
    const int64_t nn = static_cast<int64_t>(n);
    // ad - bc = n with a > b >= 0, d > c >= 0 forces a + d - 1 <= n (lem:merelpath).
    // (slot-25 HZ-REC re-audit R12) The order is (a, d, b) ascending, as before; the inner loop no longer runs over every b < a
    // (Theta(n^3) in all: about 40 min at n = 23159, the T_ell of an EXPORT read set): at t = ad - n > 0 the admissible b are the
    // divisors of t in (t/d, a) (then c = t/b < d), equivalently c | t in (t/a, d) (then b = t/c < a); the shorter of the two ranges
    // (about n/d resp. n/a terms) is scanned, the c range downwards so that b ascends.  Same set, same order, O(n^2) in all
    // (HeckeLevelNHeilbronnValidation arm (M) compares with the direct enumeration for n <= 200).
    for (int64_t a = 1; a <= nn; ++a)
        for (int64_t d = (nn + a - 1) / a; a + d - 1 <= nn; ++d) {   // t >= 0 iff d >= ceil(n / a)
            const int64_t t = a * d - nn;   // = b c >= 0
            if (t == 0) {
                for (int64_t c = 0; c < d; ++c) out.push_back(Mat22{a, 0, c, d});
                for (int64_t b = 1; b < a; ++b) out.push_back(Mat22{a, b, 0, d});
                continue;
            }
            const int64_t b_lo = t / d + 1, c_lo = t / a + 1;   // b in [b_lo, a - 1], c in [c_lo, d - 1]
            if (a - b_lo <= d - c_lo) {
                for (int64_t b = b_lo; b < a; ++b)
                    if (t % b == 0) out.push_back(Mat22{a, b, t / b, d});
            } else {
                for (int64_t c = d - 1; c >= c_lo; --c)
                    if (t % c == 0) out.push_back(Mat22{a, t / c, c, d});
            }
        }
    for (const Mat22& e : out)
        ENGINE_CHECK_MSG(e.det() == nn && e.a > e.b && e.b >= 0 && e.d > e.c && e.c >= 0,
                         "MerelSet: an element violates the defining conditions");
    return out;
}

// ---- prop:heilbronn, eq:heilbronn ------------------------------------------------------
HeilbronnOperator::HeilbronnOperator(const Gamma0LevelN& lvl, uint64_t p, uint32_t n_sym,
                                     const engine::ModCtx<uint32_t>& ctx, HeilbronnVariant variant)
    : N_(lvl.N()), p_(p), m_(lvl.m()), d_(n_sym + 1), variant_(variant), ctx_(ctx) {
    ENGINE_CHECK_MSG(p >= 2 && p < (1ull << 20) && hecke::IsPrimeU64(p), "HeilbronnOperator: p must be a prime < 2^20");
    ENGINE_CHECK_MSG(N_ < (1ull << 20), "HeilbronnOperator: N < 2^20 (Mat22 triple products, HeckeLevelN.h)");
    mer_ = MerelSet(p);
    if (variant_ == HeilbronnVariant::kTrapB)
        for (Mat22& e : mer_) e = e.adj();
    build(lvl, n_sym);
}

HeilbronnOperator::HeilbronnOperator(const Gamma0LevelN& lvl, uint64_t p, uint32_t n_sym, const engine::ModCtx<uint32_t>& ctx,
                                     std::vector<Mat22> part)
    : N_(lvl.N()), p_(p), m_(lvl.m()), d_(n_sym + 1), variant_(HeilbronnVariant::kProposition), ctx_(ctx) {
    ENGINE_CHECK_MSG(p >= 2 && p < (1ull << 20) && hecke::IsPrimeU64(p), "HeilbronnOperator: p must be a prime < 2^20");
    ENGINE_CHECK_MSG(N_ < (1ull << 20), "HeilbronnOperator: N < 2^20 (Mat22 triple products, HeckeLevelN.h)");
    for (const Mat22& e : part)
        ENGINE_CHECK_MSG(e.det() == static_cast<int64_t>(p) && e.a > e.b && e.b >= 0 && e.d > e.c && e.c >= 0,
                         "HeilbronnOperator: a partial set must lie in Mer_p");
    mer_ = std::move(part);
    build(lvl, n_sym);
}

void HeilbronnOperator::build(const Gamma0LevelN& lvl, uint32_t n_sym) {
    RhoCache rho(ctx_, n_sym);
    const bool p_divides_N = N_ % p_ == 0;
    for (const Mat22& eps : mer_) {
        const Mat22 eps_adj = eps.adj();
        for (uint32_t j = 1; j <= m_; ++j) {
            ++pairs_total_;
            if (!lvl.act_defined(j, eps)) {
                // prop:heilbronn: for p !| N every j is admissible; for p | N exactly those with
                // (c_j, d_j) eps != 0 mod p.  (det eps = p is a unit modulo the other primes of N.)
                ENGINE_CHECK_MSG(p_divides_N, "HeilbronnOperator: an inadmissible pair at p !| N (prop:heilbronn)");
                const auto pr = lvl.canonical_pair(j);
                const int64_t c2 = ModN(static_cast<int64_t>(pr.first % p_) * ModN(eps.a, p_) +
                                            static_cast<int64_t>(pr.second % p_) * ModN(eps.c, p_), p_);
                const int64_t d2 = ModN(static_cast<int64_t>(pr.first % p_) * ModN(eps.b, p_) +
                                            static_cast<int64_t>(pr.second % p_) * ModN(eps.d, p_), p_);
                ENGINE_CHECK_MSG(c2 == 0 && d2 == 0, "HeilbronnOperator: a pair dropped although (c_j, d_j) eps != 0 mod p");
                ++pairs_dropped_;
                continue;
            }
            const uint32_t i = lvl.act(j, eps);
            uint32_t blk = 0;
            // The __int128 route (slot-16 WP-13): taken on an int64 HeckeOverflow, or for every term under
            // ScopedForceWideHeilbronn (validation only, WP-17 / audit F1).
            auto wide = [&]() {
                // The int64 triple product overflows at composite N with a case-(iii) representative (entries
                // O(t N), t < c_j): e.g. N = 30030, p = 17.  The same products and the same checks in __int128
                // (Mat22W); the block is rho_n of the residues mod ctx.p -- RhoCache keys on exactly those
                // residues, so the block is the one the int64 route would have produced.
                const Mat22W h = Mat22W::from(lvl.rep(j)) * Mat22W::from(eps) * Mat22W::from(lvl.rep(i).inv_unimodular());
                const Mat22W adj_h{h.d, -h.b, -h.c, h.a};
                ENGINE_CHECK_MSG(adj_h == Mat22W::from(lvl.rep(i)) * Mat22W::from(eps_adj) *
                                              Mat22W::from(lvl.rep(j).inv_unimodular()),
                                 "HeilbronnOperator: adj h_j(eps) != w_{j.eps} adj(eps) w_j^{-1} (int128)");
                const __int128 cN = adj_h.c % static_cast<__int128>(N_);
                ENGINE_CHECK_MSG(adj_h.det() == static_cast<__int128>(p_) && cN == 0,
                                 "HeilbronnOperator: adj h_j(eps) not of det p with lower left = 0 mod N (int128)");
                const std::array<uint32_t, 4> r = (variant_ == HeilbronnVariant::kTrapA ? h : adj_h).reduced_mod(ctx_.p);
                ++wide_terms_;
                return rho.block_of(Mat22{r[0], r[1], r[2], r[3]}, blocks_);
            };
            if (t_force_wide) {
                blk = wide();
            } else try {
                const Mat22 h = lvl.cocycle(j, eps);                       // w_j eps w_i^{-1}
                const Mat22 adj_h = h.adj();
                // eq:heilbronn: adj h_j(eps) = w_{j.eps} (adj eps) w_j^{-1}; integral of det p, lower left = 0 mod N.
                ENGINE_CHECK_MSG(adj_h == lvl.rep(i) * eps_adj * lvl.rep(j).inv_unimodular(),
                                 "HeilbronnOperator: adj h_j(eps) != w_{j.eps} adj(eps) w_j^{-1}");
                ENGINE_CHECK_MSG(adj_h.det() == static_cast<int64_t>(p_) && ModN(adj_h.c, N_) == 0,
                                 "HeilbronnOperator: adj h_j(eps) not of det p with lower left = 0 mod N (lem:cosetrepsN (4))");
                const Mat22 value = variant_ == HeilbronnVariant::kTrapA ? h : adj_h;
                blk = rho.block_of(value, blocks_);
            } catch (const HeckeOverflow&) {
                blk = wide();
            }
            HeilbronnTerm t;
            t.dst = i;
            t.src = j;
            t.block = blk;
            terms_.push_back(t);
            if (lvl.rep_case(i) == 3 || lvl.rep_case(j) == 3) ++mixed_terms_;
        }
    }
    if (!p_divides_N)
        ENGINE_CHECK_MSG(pairs_dropped_ == 0, "HeilbronnOperator: p !| N but a pair was dropped");
}

std::vector<uint32_t> HeilbronnOperator::apply(const std::vector<uint32_t>& x) const {
    ENGINE_CHECK_MSG(x.size() == static_cast<size_t>(D()), "HeilbronnOperator::apply: operand length");
    const uint64_t P = ctx_.p;
    std::vector<uint64_t> acc(x.size(), 0);
    for (const HeilbronnTerm& t : terms_) {
        const Block& B = blocks_[t.block];
        const uint32_t* xs = &x[static_cast<size_t>(t.src - 1) * d_];
        uint64_t* ys = &acc[static_cast<size_t>(t.dst - 1) * d_];
        for (uint32_t r = 0; r < d_; ++r) {
            uint64_t s = 0;
            const uint32_t* brow = &B.dense[static_cast<size_t>(r) * d_];
            for (uint32_t c = 0; c < d_; ++c) s = (s + static_cast<uint64_t>(brow[c]) * xs[c]) % P;
            ys[r] = (ys[r] + s) % P;
        }
    }
    std::vector<uint32_t> y(x.size());
    for (size_t i = 0; i < y.size(); ++i) y[i] = static_cast<uint32_t>(acc[i]);
    return y;
}

std::vector<uint32_t> HeilbronnOperator::dense() const {
    const uint32_t DD = D();
    const uint64_t P = ctx_.p;
    std::vector<uint32_t> out(static_cast<size_t>(DD) * DD, 0);
    for (const HeilbronnTerm& t : terms_) {
        const Block& B = blocks_[t.block];
        for (uint32_t r = 0; r < d_; ++r)
            for (uint32_t c = 0; c < d_; ++c) {
                uint32_t& o = out[static_cast<size_t>((t.dst - 1) * d_ + r) * DD + (t.src - 1) * d_ + c];
                o = static_cast<uint32_t>((static_cast<uint64_t>(o) + B.dense[static_cast<size_t>(r) * d_ + c]) % P);
            }
    }
    return out;
}

// ---- the presentation M_N / Rel_N -------------------------------------------------------
CokernelPresentation::CokernelPresentation(const FixedSpace& fS, const FixedSpace& fU, uint32_t p) : D_(fS.D), p_(p) {
    ENGINE_CHECK_MSG(fS.D == fU.D, "CokernelPresentation: fibre ranks differ");
    ENGINE_CHECK_MSG(p >= 5 && hecke::IsPrimeU64(p), "CokernelPresentation: a prime field with 6 a unit (prop:levelorbit (2))");
    init(fS, fU);
}

CokernelPresentation::CokernelPresentation(const FixedSpace& fS, const FixedSpace& fU, uint32_t p, CurveTag) : D_(fS.D), p_(p) {
    ENGINE_CHECK_MSG(fS.D == fU.D, "CokernelPresentation: fibre ranks differ");
    ENGINE_CHECK_MSG(p >= 2 && hecke::IsPrimeU64(p), "CokernelPresentation (object=curve): a prime field");
    init(fS, fU);
}

void CokernelPresentation::init(const FixedSpace& fS, const FixedSpace& fU) {
    std::vector<std::vector<uint32_t>> rows = fS.span;
    rows.insert(rows.end(), fU.span.begin(), fU.span.end());
    // RREF over F_p.
    std::vector<int32_t> row_of_col(D_, -1);
    uint32_t rk = 0;
    for (uint32_t c = 0; c < D_ && rk < rows.size(); ++c) {
        size_t piv = rk;
        while (piv < rows.size() && rows[piv][c] == 0) ++piv;
        if (piv == rows.size()) continue;
        std::swap(rows[piv], rows[rk]);
        const uint64_t iv = InvModP(rows[rk][c], p_);
        for (uint32_t& x : rows[rk]) x = static_cast<uint32_t>(static_cast<uint64_t>(x) * iv % p_);
        for (size_t i = 0; i < rows.size(); ++i) {
            if (i == rk || rows[i][c] == 0) continue;
            const uint64_t f = rows[i][c];
            const std::vector<uint32_t>& pr = rows[rk];
            for (uint32_t k = c; k < D_; ++k)
                if (pr[k]) rows[i][k] = static_cast<uint32_t>((rows[i][k] + p_ - f * pr[k] % p_) % p_);
        }
        row_of_col[c] = static_cast<int32_t>(rk);
        pivots_.push_back(c);
        ++rk;
    }
    rows.resize(rk);
    rref_ = std::move(rows);
    for (uint32_t c = 0; c < D_; ++c)
        if (row_of_col[c] < 0) free_.push_back(c);
}

CokernelPresentation CokernelPresentation::Build(const CosetLevel& A, uint32_t n_sym, const engine::ModCtx<uint32_t>& ctx) {
    ENGINE_CHECK_MSG(ctx.is_field(), "CokernelPresentation::Build: a prime field");
    const FixedSpace fS = OrbitWiseFixedSpace(A, Mat22::S(), 2, n_sym, ctx);
    const FixedSpace fU = OrbitWiseFixedSpace(A, Mat22::ST(), 3, n_sym, ctx);
    return CokernelPresentation(fS, fU, ctx.p);
}

CokernelPresentation CokernelPresentation::BuildCurve(const CosetLevel& A, uint32_t n_sym, const engine::ModCtx<uint32_t>& ctx) {
    ENGINE_CHECK_MSG(n_sym == 0, "CokernelPresentation::BuildCurve: object=curve is the weight-2 (n = 0) presentation only "
                                 "(LN23_spec L2); n > 0 refused");
    ENGINE_CHECK_MSG(ctx.is_field(), "CokernelPresentation::BuildCurve: a prime field");
    const FixedSpace fS = OrbitWiseFixedSpace(A, Mat22::S(), 2, 0, ctx);
    const FixedSpace fU = OrbitWiseFixedSpace(A, Mat22::ST(), 3, 0, ctx);
    return CokernelPresentation(fS, fU, ctx.p, CurveTag{});
}

std::vector<uint32_t> CokernelPresentation::coords(const std::vector<uint32_t>& x) const {
    ENGINE_CHECK_MSG(x.size() == D_, "CokernelPresentation::coords: operand length");
    std::vector<uint64_t> w(x.begin(), x.end());
    for (uint64_t& v : w) v %= p_;
    // Rows are fully reduced (zero in every other pivot column), so subtracting x[pivot] * row
    // for each row, with the ORIGINAL x[pivot], reduces x to its free-column normal form.
    for (size_t r = 0; r < rref_.size(); ++r) {
        const uint64_t f = w[pivots_[r]];
        if (f == 0) continue;
        const std::vector<uint32_t>& row = rref_[r];
        for (uint32_t k = pivots_[r]; k < D_; ++k)
            if (row[k]) w[k] = (w[k] + p_ - f * row[k] % p_) % p_;
    }
    std::vector<uint32_t> out(free_.size());
    for (size_t i = 0; i < free_.size(); ++i) out[i] = static_cast<uint32_t>(w[free_[i]]);
    return out;
}

bool CokernelPresentation::in_rel(const std::vector<uint32_t>& x) const {
    const std::vector<uint32_t> c = coords(x);
    return std::all_of(c.begin(), c.end(), [](uint32_t v) { return v == 0; });
}

uint32_t RelEscapes(const CokernelPresentation& pres, const HeilbronnOperator& H) {
    ENGINE_CHECK_MSG(H.ctx().is_field() && H.ctx().p == pres.p() && H.D() == pres.D(),
                     "RelEscapes: operator and presentation over the same field and fibre");
    uint32_t bad = 0;
    for (const auto& r : pres.rel_basis()) bad += pres.in_rel(H.apply(r)) ? 0 : 1;
    return bad;
}

HMat OnCokernel(const CokernelPresentation& pres, const HeilbronnOperator& H) {
    ENGINE_CHECK_MSG(RelEscapes(pres, H) == 0, "OnCokernel: H_{N,p}(Rel_N) is not contained in Rel_N (prop:heilbronn (1))");
    const uint32_t h = pres.h1();
    HMat M;
    M.rows = h;
    M.cols = h;
    M.modulus = pres.p();
    M.ell = pres.p();
    M.field = true;
    M.row_exp.assign(h, 1);
    M.a.assign(static_cast<size_t>(h) * h, 0);
    std::vector<uint32_t> e(pres.D(), 0);
    for (uint32_t j = 0; j < h; ++j) {
        e[pres.free_columns()[j]] = 1;
        const std::vector<uint32_t> c = pres.coords(H.apply(e));
        e[pres.free_columns()[j]] = 0;
        for (uint32_t i = 0; i < h; ++i) M.a[static_cast<size_t>(i) * h + j] = c[i];
    }
    return M;
}

} // namespace level
} // namespace hecke
