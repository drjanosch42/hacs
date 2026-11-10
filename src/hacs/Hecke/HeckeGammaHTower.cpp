// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// (hecke-extensions slot 22, P10) The character model on Gamma_H(N) over Z/ell^K -- see HeckeGammaHTower.h and
// docs/implementation/slot22/P10_spec.md.
#include "HeckeGammaHTower.h"

#include <algorithm>
#include <chrono>
#include <numeric>
#include <cstdlib>
#include <stdexcept>
#include <thread>

#include "HeckeLevelNHeilbronn.h"   // MerelSet
#include "../../ModArithmetic/EngineCheck.h"
#include "../../ModArithmetic/ScalarMod.h"
#include "../../Engine/Threads.h"

namespace hecke {
namespace level {
namespace tower {

namespace {

using engine::scalar::gcd;
using engine::scalar::modi;
using engine::scalar::mulmod;
using engine::scalar::powmod;

using u128 = unsigned __int128;
using M4 = std::array<uint64_t, 4>;   // a 2x2 matrix mod N, row-major

uint64_t InvModU(uint64_t a, uint64_t m) {
    uint64_t x = 0;
    ENGINE_CHECK_MSG(engine::scalar::try_invmod<uint64_t>(a, m, x), "tower: an inverse of a non-unit mod N");
    return x;
}
M4 Red(const Mat22& g, uint64_t N) { return {modi(g.a, N), modi(g.b, N), modi(g.c, N), modi(g.d, N)}; }
M4 AdjRed(const Mat22& g, uint64_t N) { return {modi(g.d, N), modi(-g.b, N), modi(-g.c, N), modi(g.a, N)}; }
M4 Mul(const M4& x, const M4& y, uint64_t N) {
    return {(mulmod(x[0], y[0], N) + mulmod(x[1], y[2], N)) % N, (mulmod(x[0], y[1], N) + mulmod(x[1], y[3], N)) % N,
            (mulmod(x[2], y[0], N) + mulmod(x[3], y[2], N)) % N, (mulmod(x[2], y[1], N) + mulmod(x[3], y[3], N)) % N};
}

// [begin, end) split into contiguous chunks, one per worker; fn(begin, end, worker).
template <class F>
void ParallelFor(uint64_t n, F fn) {
    const uint32_t T = static_cast<uint32_t>(std::max<uint64_t>(1, std::min<uint64_t>(TowerThreads(), n / 16 + 1)));
    if (T <= 1) { fn(0, n, 0u); return; }
    std::vector<std::thread> th;
    for (uint32_t w = 0; w < T; ++w) {
        const uint64_t b = n * w / T, e = n * (w + 1) / T;
        th.emplace_back([&fn, b, e, w]() { fn(b, e, w); });
    }
    for (auto& t : th) t.join();
}

} // namespace

uint32_t TowerThreads() { return engine::threads::configured(engine::threads::kGammaHTower); }

// ============================================================================
// ZK
// ============================================================================
ZK::ZK(uint32_t ell_, uint32_t K_) : ell(ell_), K(K_), Q(1) {
    ENGINE_CHECK_MSG(ell_ >= 2 && K_ >= 1, "ZK: ell >= 2, K >= 1");
    for (uint32_t i = 0; i < K_; ++i) {
        Q *= ell_;
        ENGINE_CHECK_MSG(Q < (1ull << 32), "ZK: ell^K < 2^32");
    }
}
uint32_t ZK::pow(uint32_t a, uint64_t e) const {
    uint64_t r = 1 % Q, b = a % Q;
    while (e) {
        if (e & 1) r = r * b % Q;
        b = b * b % Q;
        e >>= 1;
    }
    return static_cast<uint32_t>(r);
}
uint32_t ZK::inv(uint32_t a) const {
    ENGINE_CHECK_MSG(unit(a), "ZK::inv: not a unit");
    return static_cast<uint32_t>(InvModU(a, Q));
}
uint32_t ZK::val(uint32_t a) const {
    if (a % Q == 0) return K;
    uint32_t v = 0;
    while (a % ell == 0) { a /= ell; ++v; }
    return v;
}
uint32_t ZK::from_i64(int64_t x) const { return static_cast<uint32_t>(modi(x, Q)); }
uint32_t ZK::teich(uint64_t x) const {
    ENGINE_CHECK_MSG(x % ell != 0, "ZK::teich: x not a unit mod ell");
    uint32_t t = static_cast<uint32_t>(x % Q);
    for (uint32_t i = 0; i < K + 1; ++i) t = pow(t, ell);
    return t;
}

std::vector<uint32_t> TameCharacterTable(uint64_t N, const ZK& R, int64_t e, bool trivial) {
    std::vector<uint32_t> t(N, 0);
    const int64_t lm1 = static_cast<int64_t>(R.ell) - 1;
    const uint64_t ee = static_cast<uint64_t>(((e % lm1) + lm1) % lm1);
    for (uint64_t x = 1; x < N; ++x) {
        if (gcd<uint64_t>(x, N) != 1) continue;
        t[x] = trivial ? static_cast<uint32_t>(1 % R.Q) : R.pow(R.teich(x), ee);
    }
    if (N == 1) t[0] = static_cast<uint32_t>(1 % R.Q);
    return t;
}

// ============================================================================
// monomial matrices and the relation generators
// ============================================================================
MonomialChi MonomialChiOf(const GammaHLevelN& G, const Mat22& gamma, const std::vector<uint32_t>& chi) {
    const uint64_t N = G.N();
    const uint32_t m = G.m();
    ENGINE_CHECK_MSG(chi.size() == N, "MonomialChiOf: the character table has size N");
    MonomialChi M;
    M.tgt.assign(m + 1, 0);
    M.scal.assign(m + 1, 0);
    M.src.assign(m + 1, 0);
    for (uint32_t J = 1; J <= m; ++J) {
        const Mat22 x = G.rep(J) * gamma;
        const uint32_t Jg = G.class_of(x);
        const M4 h = Mul(Red(x, N), AdjRed(G.rep(Jg), N), N);
        ENGINE_CHECK_MSG(h[2] == 0 && G.in_H(h[3]) && mulmod(h[0], h[3], N) == 1 % N,
                         "MonomialChiOf: h_J(gamma) is not in Gamma_H(N) mod N");
        M.tgt[J] = Jg;
        M.scal[J] = chi[h[0]];
        ENGINE_CHECK_MSG(M.src[Jg] == 0, "MonomialChiOf: J -> J.gamma is not a permutation");
        M.src[Jg] = J;
    }
    return M;
}

void RelGeneratorsChi(const MonomialChi& S, const MonomialChi& U, const ZK& R, uint32_t ord_S, uint32_t ord_U,
                      std::vector<SparseVec>& gen_S, std::vector<SparseVec>& gen_U) {
    const uint32_t m = static_cast<uint32_t>(S.tgt.size() - 1);
    std::vector<std::pair<uint32_t, uint32_t>> e;
    auto push = [&](std::vector<SparseVec>& out) {
        std::sort(e.begin(), e.end());
        SparseVec v;
        for (size_t i = 0; i < e.size();) {
            uint64_t s = 0;
            size_t j = i;
            for (; j < e.size() && e[j].first == e[i].first; ++j) s += e[j].second;
            s %= R.Q;
            if (s) { v.col.push_back(e[i].first); v.val.push_back(static_cast<uint32_t>(s)); }
            i = j;
        }
        if (!v.col.empty()) out.push_back(std::move(v));
    };
    // M(gamma) e_K = scal[J] e_J, J = src[K]; the norm sum_{k < ord} M(gamma)^k e_K, with M(gamma)^ord e_K = e_K checked.
    auto norm = [&](const MonomialChi& M, uint32_t ord, std::vector<SparseVec>& out) {
        for (uint32_t K = 1; K <= m; ++K) {
            e.clear();
            uint32_t cur = K, c = static_cast<uint32_t>(1 % R.Q);
            for (uint32_t k = 0; k < ord; ++k) {
                e.push_back({cur - 1, c});
                const uint32_t J = M.src[cur];
                c = R.mul(c, M.scal[J]);
                cur = J;
            }
            ENGINE_CHECK_MSG(cur == K && c == 1 % R.Q, "RelGeneratorsChi: M(gamma)^ord != 1 (the order or the parity is wrong)");
            push(out);
        }
    };
    gen_S.clear();
    gen_U.clear();
    norm(S, ord_S, gen_S);
    norm(U, ord_U, gen_U);
}

// ============================================================================
// ChainCokernel
// ============================================================================
ChainCokernel::ChainCokernel(uint32_t D, std::vector<SparseVec> gens, const ZK& R, uint64_t byte_cap, bool allow_torsion) : D_(D), R_(R) {
    const auto t0 = std::chrono::steady_clock::now();
    const uint64_t Q = R.Q;
    st_.generators = gens.size();
    // the shipped order (HeckeLevelNSparse, slot-18 WP-5 (b)): largest column descending, stable
    std::stable_sort(gens.begin(), gens.end(), [](const SparseVec& a, const SparseVec& b) {
        const uint32_t la = a.col.empty() ? 0 : a.col.back(), lb = b.col.empty() ? 0 : b.col.back();
        return la > lb;
    });
    std::vector<SparseVec> rows;
    std::vector<int32_t> row_of_col(D, -1);
    std::vector<std::vector<uint32_t>> occ(D);
    std::vector<SparseVec> residuals;
    std::vector<uint64_t> acc(D, 0);
    std::vector<uint8_t> mark(D, 0);
    std::vector<uint32_t> touched;
    uint64_t nnz = 0, occ_entries = 0;

    // x reduced against the current rows (rows carry no pivot column but their own): dense scatter.
    auto reduce = [&](const SparseVec& g, SparseVec& outv) {
        touched.clear();
        auto bump = [&](uint32_t c, uint64_t v) {
            if (!mark[c]) { mark[c] = 1; touched.push_back(c); acc[c] = 0; }
            acc[c] = (acc[c] + v) % Q;
        };
        for (size_t k = 0; k < g.col.size(); ++k) {
            const uint32_t c = g.col[k];
            const uint64_t v = g.val[k] % Q;
            if (!v) continue;
            const int32_t rid = row_of_col[c];
            if (rid < 0) { bump(c, v); continue; }
            const SparseVec& r = rows[static_cast<size_t>(rid)];
            const uint64_t mv = Q - v;
            for (size_t t = 0; t < r.col.size(); ++t) bump(r.col[t], mv * r.val[t] % Q);
            st_.merge_ops += r.col.size();
        }
        std::sort(touched.begin(), touched.end());
        outv.col.clear();
        outv.val.clear();
        for (uint32_t c : touched) {
            mark[c] = 0;
            if (acc[c] % Q) { outv.col.push_back(c); outv.val.push_back(static_cast<uint32_t>(acc[c] % Q)); }
        }
    };

    SparseVec red, merged;
    for (const SparseVec& g : gens) {
        reduce(g, red);
        if (red.col.empty()) { ++st_.dependent; continue; }
        size_t piv = red.col.size();
        for (size_t k = 0; k < red.col.size(); ++k)
            if (R.unit(red.val[k])) { piv = k; break; }
        if (piv == red.col.size()) { ++st_.residual_nonunit; residuals.push_back(red); continue; }
        const uint32_t cstar = red.col[piv];
        const uint32_t inv = R.inv(red.val[piv]);
        SparseVec tail;
        for (size_t k = 0; k < red.col.size(); ++k) {
            if (k == piv) continue;
            tail.col.push_back(red.col[k]);
            tail.val.push_back(R.mul(red.val[k], inv));
        }
        // Gauss--Jordan: clear cstar from every existing row that carries it
        for (uint32_t rid : occ[cstar]) {
            SparseVec& r = rows[rid];
            const auto it = std::lower_bound(r.col.begin(), r.col.end(), cstar);
            if (it == r.col.end() || *it != cstar) continue;   // stale occurrence
            const size_t pos = static_cast<size_t>(it - r.col.begin());
            const uint32_t f = r.val[pos];
            merged.col.clear();
            merged.val.clear();
            size_t a = 0, b = 0;
            const uint64_t mf = Q - f;
            while (a < r.col.size() || b < tail.col.size()) {
                if (a == pos) { ++a; continue; }
                if (b >= tail.col.size() || (a < r.col.size() && r.col[a] < tail.col[b])) {
                    merged.col.push_back(r.col[a]); merged.val.push_back(r.val[a]); ++a;
                } else if (a >= r.col.size() || tail.col[b] < r.col[a]) {
                    const uint32_t v = static_cast<uint32_t>(mf * tail.val[b] % Q);
                    if (v) { merged.col.push_back(tail.col[b]); merged.val.push_back(v); occ[tail.col[b]].push_back(rid); ++occ_entries; }
                    ++b;
                } else {
                    const uint32_t v = static_cast<uint32_t>((r.val[a] + mf * tail.val[b]) % Q);
                    if (v) { merged.col.push_back(r.col[a]); merged.val.push_back(v); }
                    ++a; ++b;
                }
            }
            st_.merge_ops += r.col.size() + tail.col.size();
            nnz = nnz - r.col.size() + merged.col.size();
            std::swap(r, merged);
        }
        occ_entries -= occ[cstar].size();
        std::vector<uint32_t>().swap(occ[cstar]);
        const uint32_t nid = static_cast<uint32_t>(rows.size());
        for (uint32_t c : tail.col) { occ[c].push_back(nid); ++occ_entries; }
        nnz += tail.col.size();
        rows.push_back(std::move(tail));
        row_of_col[cstar] = static_cast<int32_t>(nid);
        st_.peak_nnz = std::max(st_.peak_nnz, nnz);
        const uint64_t bytes = nnz * 8 + occ_entries * 4 + rows.size() * sizeof(SparseVec);
        st_.peak_bytes = std::max(st_.peak_bytes, bytes);
        if (byte_cap > 0 && bytes > byte_cap)
            throw std::logic_error("tower_rss_cap: the chain-ring elimination stores " + std::to_string(bytes) +
                                   " bytes > the cap " + std::to_string(byte_cap));
    }
    // every residual must lie in the span of the final rows (lem:nebchainrref: the quotient is free)
    std::vector<SparseVec> kept;
    for (const SparseVec& r : residuals) {
        reduce(r, red);
        if (allow_torsion) { if (!red.col.empty()) kept.push_back(red); continue; }
        ENGINE_CHECK_MSG(red.col.empty(), "chainring_pivot: a residual without a unit entry is not absorbed by the unit-pivot rows "
                                          "(the quotient is not free over Z/ell^K)");
    }
    std::vector<std::vector<uint32_t>>().swap(occ);
    // pivots, free columns, tails in free-index space
    free_index_.assign(D, -1);
    tail_of_.assign(D, -1);
    for (uint32_t c = 0; c < D; ++c) {
        if (row_of_col[c] >= 0) pivots_.push_back(c);
        else { free_index_[c] = static_cast<int32_t>(free_.size()); free_.push_back(c); }
    }
    tails_.resize(pivots_.size());
    for (size_t k = 0; k < pivots_.size(); ++k) {
        const uint32_t c = pivots_[k];
        const SparseVec& r = rows[static_cast<size_t>(row_of_col[c])];
        SparseVec t;
        t.col.reserve(r.col.size());
        t.val.reserve(r.col.size());
        for (size_t i = 0; i < r.col.size(); ++i) {
            const int32_t f = free_index_[r.col[i]];
            ENGINE_CHECK_MSG(f >= 0, "ChainCokernel: a final row carries a pivot column");
            t.col.push_back(static_cast<uint32_t>(f));
            t.val.push_back(r.val[i]);
        }
        tail_of_[c] = static_cast<int32_t>(k);
        st_.final_nnz += t.col.size();
        tails_[k] = std::move(t);
    }
    for (const SparseVec& r : kept) {
        SparseVec t;
        for (size_t i = 0; i < r.col.size(); ++i) {
            const int32_t f = free_index_[r.col[i]];
            ENGINE_CHECK_MSG(f >= 0, "ChainCokernel: a reduced residual at a pivot column");
            t.col.push_back(static_cast<uint32_t>(f));
            t.val.push_back(r.val[i]);
        }
        torsion_.push_back(std::move(t));
    }
    st_.seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
}

void ChainCokernel::add_coords(uint32_t col, uint32_t coef, std::vector<uint32_t>& out) const {
    if (!coef) return;
    const int32_t f = free_index_[col];
    if (f >= 0) { out[static_cast<size_t>(f)] = R_.add(out[static_cast<size_t>(f)], coef); return; }
    const SparseVec& t = tails_[static_cast<size_t>(tail_of_[col])];
    const uint32_t mc = R_.neg(coef);
    for (size_t i = 0; i < t.col.size(); ++i) out[t.col[i]] = R_.add(out[t.col[i]], R_.mul(mc, t.val[i]));
}

std::vector<uint32_t> ChainCokernel::coords(const std::vector<uint32_t>& x) const {
    ENGINE_CHECK_MSG(x.size() == D_, "ChainCokernel::coords: length D");
    std::vector<uint32_t> out(h(), 0);
    for (uint32_t c = 0; c < D_; ++c) add_coords(c, x[c], out);
    return out;
}

bool ChainCokernel::in_rel(const SparseVec& x) const {
    std::vector<uint32_t> out(h(), 0);
    for (size_t k = 0; k < x.col.size(); ++k) add_coords(x.col[k], x.val[k] % static_cast<uint32_t>(R_.Q), out);
    for (uint32_t v : out) if (v) return false;
    return true;
}

// ============================================================================
// operators
// ============================================================================
ChiHeilbronn::ChiHeilbronn(const GammaHLevelN& G, uint64_t n, const std::vector<uint32_t>& chi, const ZK& R)
    : G_(&G), n_(n), chi_(&chi), R_(R) {
    ENGINE_CHECK_MSG(n >= 1 && n < (1ull << 20), "ChiHeilbronn: n in [1, 2^20)");
    ENGINE_CHECK_MSG(chi.size() == G.N(), "ChiHeilbronn: the character table has size N");
    mer_ = MerelSet(n);
    repN_.assign(G.m() + 1, {0, 0, 0, 0});
    adjN_.assign(G.m() + 1, {0, 0, 0, 0});
    for (uint32_t J = 1; J <= G.m(); ++J) {
        const Mat22 w = G.rep(J);
        repN_[J] = Red(w, G.N());
        adjN_[J] = AdjRed(w, G.N());
    }
}

std::string ChiHeilbronn::label() const { return std::string(gcd<uint64_t>(n_, G_->N()) == 1 ? "T_" : "U_") + std::to_string(n_); }

void ChiHeilbronn::image(uint32_t J, std::vector<std::pair<uint32_t, uint32_t>>& out) const {
    const uint64_t N = G_->N();
    const bool coprime = gcd<uint64_t>(n_, N) == 1;
    for (const Mat22& eps : mer_) {
        if (!G_->act_defined(J, eps)) {
            ENGINE_CHECK_MSG(!coprime, "ChiHeilbronn: an inadmissible pair at (n, N) = 1");
            continue;
        }
        const uint32_t i = G_->act(J, eps);
        const M4 h = Mul(Mul(repN_[J], Red(eps, N), N), adjN_[i], N);
        ENGINE_CHECK_MSG(h[2] == 0 && G_->in_H(h[3]) && mulmod(h[0], h[3], N) == n_ % N,
                         "ChiHeilbronn: h_J(eps) is not upper triangular mod N with d_h in H and det n");
        out.push_back({i, (*chi_)[h[3]]});
    }
}

ChiDiamond::ChiDiamond(const GammaHLevelN& G, uint64_t dd, const std::vector<uint32_t>& chi, const ZK& R) : d_(dd) {
    const uint64_t N = G.N();
    const uint32_t m = G.m();
    ENGINE_CHECK_MSG(chi.size() == N && gcd<uint64_t>(dd % N, N) == 1, "ChiDiamond: d a unit mod N, chi of size N");
    const uint64_t d = dd % N, dinv = InvModU(d, N);
    std::vector<uint32_t> src(m + 1, 0);
    std::vector<M4> RJ(m + 1);
    for (uint32_t J = 1; J <= m; ++J) {
        const uint32_t j = G.j_of(J);
        const uint64_t u = G.coset_reps()[G.r_of(J)];
        const uint32_t r2 = G.coset_rank(mulmod(dinv, u, N));
        src[J] = G.index(j, r2);
        const Mat22 Rm = G.sigma(u) * G.sigma(G.coset_reps()[r2]).inv_unimodular();
        RJ[J] = Red(Rm, N);
        ENGINE_CHECK_MSG(RJ[J][2] == 0 && G.in_H(mulmod(RJ[J][3], dinv, N)), "ChiDiamond: R_J not in Gamma_0(N) with d_R in d H");
    }
    // the fibre-map identity mod N for S and ST
    for (const Mat22& gam : {Mat22::S(), Mat22::ST()}) {
        for (uint32_t J = 1; J <= m; ++J) {
            const Mat22 x = G.rep(J) * gam;
            const uint32_t Jg = G.class_of(x);
            const M4 hJ = Mul(Red(x, N), AdjRed(G.rep(Jg), N), N);
            const uint32_t Js = src[J];
            const Mat22 y = G.rep(Js) * gam;
            const uint32_t Jsg = G.class_of(y);
            const M4 hs = Mul(Red(y, N), AdjRed(G.rep(Jsg), N), N);
            ENGINE_CHECK_MSG(src[Jg] == Jsg && Mul(hJ, RJ[Jg], N) == Mul(RJ[J], hs, N),
                             "ChiDiamond: <d> does not commute with the Gamma-action (mod N)");
        }
    }
    dst_.assign(m + 1, 0);
    scal_.assign(m + 1, 0);
    for (uint32_t J = 1; J <= m; ++J) {
        ENGINE_CHECK_MSG(dst_[src[J]] == 0, "ChiDiamond: not a permutation of the classes");
        dst_[src[J]] = J;
        scal_[src[J]] = chi[RJ[J][0]];
    }
    (void)R;
}

void ChiDiamond::image(uint32_t J, std::vector<std::pair<uint32_t, uint32_t>>& out) const { out.push_back({dst_[J], scal_[J]}); }

void QuotientOperator::apply(const ZK& R, const std::vector<uint32_t>& x, std::vector<uint32_t>& y) const {
    y.assign(h, 0);
    std::vector<uint64_t> acc(h, 0);
    for (uint32_t f = 0; f < h; ++f) {
        const uint64_t xf = x[f];
        if (!xf) continue;
        for (uint64_t k = start[f]; k < start[f + 1]; ++k) acc[ent[k].first] = (acc[ent[k].first] + xf * ent[k].second) % R.Q;
    }
    for (uint32_t i = 0; i < h; ++i) y[i] = static_cast<uint32_t>(acc[i]);
}

void QuotientOperator::apply_block(const ZK& R, const std::vector<uint32_t>& B, uint32_t s, std::vector<uint32_t>& Y) const {
    Y.assign(static_cast<size_t>(h) * s, 0);
    ParallelFor(s, [&](uint64_t kb, uint64_t ke, uint32_t) {
        std::vector<uint64_t> acc(h);
        for (uint64_t k = kb; k < ke; ++k) {
            std::fill(acc.begin(), acc.end(), 0);
            const uint32_t* x = &B[static_cast<size_t>(k) * h];
            for (uint32_t f = 0; f < h; ++f) {
                const uint64_t xf = x[f];
                if (!xf) continue;
                for (uint64_t t = start[f]; t < start[f + 1]; ++t) {
                    uint64_t& a = acc[ent[t].first];
                    a += xf * ent[t].second;
                    if (a >= (1ull << 62)) a %= R.Q;
                }
            }
            uint32_t* y = &Y[static_cast<size_t>(k) * h];
            for (uint32_t i = 0; i < h; ++i) y[i] = static_cast<uint32_t>(acc[i] % R.Q);
        }
    });
}

QuotientOperator OnQuotient(const ChainCokernel& Q, const ChiOperator& X) {
    QuotientOperator op;
    op.h = Q.h();
    op.label = X.label();
    op.start.assign(static_cast<size_t>(op.h) + 1, 0);
    // columns in parallel (chunk-local entry lists, concatenated in column order: the result is the serial one)
    const uint32_t T = TowerThreads();
    std::vector<std::vector<std::pair<uint32_t, uint32_t>>> part(T);
    std::vector<uint64_t> cnt(op.h, 0);
    ParallelFor(op.h, [&](uint64_t fb, uint64_t fe, uint32_t w) {
        std::vector<uint32_t> acc(op.h, 0);
        std::vector<uint8_t> mark(op.h, 0);
        std::vector<uint32_t> touched;
        std::vector<std::pair<uint32_t, uint32_t>> img;
        std::vector<uint32_t> one;
        for (uint64_t f = fb; f < fe; ++f) {
            img.clear();
            X.image(Q.free_columns()[f] + 1, img);
            touched.clear();
            for (const auto& e : img) {
                const uint32_t col = e.first - 1;
                const int32_t fi = Q.free_index(col);
                auto bump = [&](uint32_t i, uint32_t v) {
                    if (!mark[i]) { mark[i] = 1; touched.push_back(i); }
                    acc[i] = Q.ring().add(acc[i], v);
                };
                if (fi >= 0) { bump(static_cast<uint32_t>(fi), e.second); continue; }
                const SparseVec& t = Q.tail(col);
                const uint32_t mc = Q.ring().neg(e.second);
                for (size_t i = 0; i < t.col.size(); ++i) bump(t.col[i], Q.ring().mul(mc, t.val[i]));
            }
            std::sort(touched.begin(), touched.end());
            uint64_t c = 0;
            for (uint32_t i : touched) {
                if (acc[i]) { part[w].push_back({i, acc[i]}); ++c; }
                acc[i] = 0;
                mark[i] = 0;
            }
            cnt[f] = c;
        }
    });
    uint64_t total = 0;
    for (uint32_t f = 0; f < op.h; ++f) { op.start[f] = total; total += cnt[f]; }
    op.start[op.h] = total;
    op.ent.reserve(total);
    for (uint32_t w = 0; w < T; ++w) op.ent.insert(op.ent.end(), part[w].begin(), part[w].end());
    ENGINE_CHECK_MSG(op.ent.size() == total, "OnQuotient: entry count");
    return op;
}

uint32_t RelEscapesChi(const ChainCokernel& Q, const std::vector<SparseVec>& gens, const ChiOperator& X) {
    const ZK& R = Q.ring();
    std::vector<uint32_t> badw(TowerThreads(), 0);
    ParallelFor(gens.size(), [&](uint64_t gb, uint64_t ge, uint32_t w) {
        std::vector<uint32_t> out(Q.h(), 0);
        std::vector<std::pair<uint32_t, uint32_t>> img;
        for (uint64_t gi = gb; gi < ge; ++gi) {
            const SparseVec& g = gens[gi];
            std::fill(out.begin(), out.end(), 0);
            for (size_t k = 0; k < g.col.size(); ++k) {
                img.clear();
                X.image(g.col[k] + 1, img);
                for (const auto& e : img) Q.add_coords(e.first - 1, R.mul(e.second, g.val[k]), out);
            }
            for (uint32_t v : out)
                if (v) { ++badw[w]; break; }
        }
    });
    uint32_t bad = 0;
    for (uint32_t b : badw) bad += b;
    return bad;
}

HowellZK::HowellZK(const ZK& R, std::vector<std::vector<uint32_t>> rest, uint32_t m) : R_(R) {
    auto ipow = [&](uint32_t e) { uint64_t r = 1; while (e--) r *= R.ell; return static_cast<uint32_t>(r); };
    for (uint32_t c = 0; c < m; ++c) {
        size_t best = rest.size();
        uint32_t bv = R.K;
        for (size_t i = 0; i < rest.size(); ++i) {
            const uint32_t v = R.val(rest[i][c]);
            if (v < bv) { bv = v; best = i; if (v == 0) break; }
        }
        if (best == rest.size()) continue;
        std::vector<uint32_t> pr = std::move(rest[best]);
        rest.erase(rest.begin() + static_cast<long>(best));
        const uint32_t pe = ipow(bv);
        const uint32_t unit = R.inv(static_cast<uint32_t>((pr[c] / pe) % R.Q));
        for (auto& x : pr) x = R.mul(x, unit);
        auto clear = [&](std::vector<uint32_t>& x) {
            if (!x[c]) return;
            const uint32_t f = x[c] / pe;
            for (uint32_t j = 0; j < m; ++j) if (pr[j]) x[j] = R.sub(x[j], R.mul(f, pr[j]));
        };
        for (auto& x : rest) clear(x);
        for (auto& x : rows_) clear(x);
        if (bv > 0) {
            std::vector<uint32_t> an(m);
            const uint32_t f = ipow(R.K - bv) % static_cast<uint32_t>(R.Q);
            bool nz = false;
            for (uint32_t j = 0; j < m; ++j) { an[j] = R.mul(f, pr[j]); nz = nz || an[j]; }
            if (nz) rest.push_back(std::move(an));
        }
        // drop rows that became zero
        rest.erase(std::remove_if(rest.begin(), rest.end(), [](const std::vector<uint32_t>& x) {
                       for (uint32_t v : x) if (v) return false;
                       return true;
                   }), rest.end());
        rows_.push_back(std::move(pr));
        pcol_.push_back(c);
        pval_.push_back(bv);
        length_ += R.K - bv;
    }
}

bool HowellZK::contains(std::vector<uint32_t> x) const {
    for (size_t k = 0; k < rows_.size(); ++k) {
        const uint32_t c = pcol_[k];
        if (!x[c]) continue;
        uint64_t pe = 1;
        for (uint32_t i = 0; i < pval_[k]; ++i) pe *= R_.ell;
        if (x[c] % pe) return false;
        const uint32_t f = static_cast<uint32_t>(x[c] / pe);
        for (size_t j = 0; j < x.size(); ++j) if (rows_[k][j]) x[j] = R_.sub(x[j], R_.mul(f, rows_[k][j]));
    }
    for (uint32_t v : x) if (v) return false;
    return true;
}

} // namespace tower
} // namespace level
} // namespace hecke
