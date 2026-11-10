// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// (hecke-extensions slot 18, WP-6) The character model over W_K(F_q) -- see HeckeNebentypusGalois.h
// ([C] sec:nebentypus: lem:teichmuller, prop:neborbit (1), (3), rmk:nebcost (4), code map item (iv)).
#include "HeckeNebentypusGalois.h"

#include <algorithm>
#include <chrono>
#include <map>
#include <sstream>
#include <utility>

#include "HeckeLevelEll.h"
#include "../../ModArithmetic/EngineCheck.h"
#include "../../ModArithmetic/ModCtx.h"
#include "../Util/Timing.h"

namespace hecke {
namespace neb {

namespace {

using oc_util::Seconds;

using u128 = unsigned __int128;

}  // namespace

// ============================================================================
// WkArith
// ============================================================================
WkArith::WkArith(const Fq& F, uint32_t K) : F_(&F), K_(K), d_(F.degree()), ell_(F.ell()) {
    if (K < 1) throw NebentypusRefusal("wk_ring_too_large", "K >= 1");
    if (d_ > kWkMaxDegree)
        throw NebentypusRefusal("wk_ring_too_large", "d = " + std::to_string(d_) + " > " + std::to_string(kWkMaxDegree));
    pw_.assign(K + 1, 1);
    for (uint32_t j = 1; j <= K; ++j) {
        pw_[j] = pw_[j - 1] * ell_;
        if (pw_[j] >= (1ull << 32))
            throw NebentypusRefusal("wk_ring_too_large", "ell^K >= 2^32 (ell = " + std::to_string(ell_) + ", K = " + std::to_string(K) + ")");
    }
    C_.assign(d_ + 1, 0);
    for (uint32_t i = 0; i <= d_; ++i) C_[i] = F.defining_poly().coeff(i);
    ENGINE_CHECK_MSG(C_[d_] == 1, "WkArith: the defining polynomial is monic");
}

Wel WkArith::scalar(int64_t s) const {
    const int64_t m = static_cast<int64_t>(pw_[K_]);
    int64_t r = s % m;
    if (r < 0) r += m;
    Wel w{};
    w[0] = static_cast<uint64_t>(r);
    return w;
}

Wel WkArith::lift(Fq::E x) const {
    const std::vector<uint32_t> c = F_->digits(x);
    Wel w{};
    for (uint32_t i = 0; i < d_; ++i) w[i] = c[i];
    return w;
}

Fq::E WkArith::reduce(const Wel& a) const {
    std::vector<uint32_t> c(d_);
    for (uint32_t i = 0; i < d_; ++i) c[i] = static_cast<uint32_t>(a[i] % ell_);
    return F_->from_digits(c);
}

bool WkArith::is_zero(const Wel& a) const {
    for (uint32_t i = 0; i < d_; ++i) if (a[i]) return false;
    return true;
}

Wel WkArith::add(const Wel& a, const Wel& b) const {
    Wel w{};
    const uint64_t m = pw_[K_];
    for (uint32_t i = 0; i < d_; ++i) { const uint64_t s = a[i] + b[i]; w[i] = s >= m ? s - m : s; }
    return w;
}

Wel WkArith::sub(const Wel& a, const Wel& b) const {
    Wel w{};
    const uint64_t m = pw_[K_];
    for (uint32_t i = 0; i < d_; ++i) w[i] = a[i] >= b[i] ? a[i] - b[i] : a[i] + m - b[i];
    return w;
}

Wel WkArith::mul(const Wel& a, const Wel& b) const {
    const uint64_t m = pw_[K_];
    if (d_ == 1) { Wel w{}; w[0] = a[0] * b[0] % m; return w; }   // < 2^64: both factors < 2^32
    uint64_t z[2 * kWkMaxDegree - 1] = {0};
    for (uint32_t i = 0; i < d_; ++i) {
        if (!a[i]) continue;
        for (uint32_t j = 0; j < d_; ++j) z[i + j] = (z[i + j] + a[i] * b[j] % m) % m;
    }
    for (uint32_t k = 2 * d_ - 1; k-- > d_;) {
        const uint64_t t = z[k];
        if (!t) continue;
        z[k] = 0;
        for (uint32_t i = 0; i < d_; ++i) z[k - d_ + i] = (z[k - d_ + i] + m - t * C_[i] % m) % m;
    }
    Wel w{};
    for (uint32_t i = 0; i < d_; ++i) w[i] = z[i];
    return w;
}

Wel WkArith::mul_scalar(const Wel& a, uint64_t s) const {
    const uint64_t m = pw_[K_];
    Wel w{};
    for (uint32_t i = 0; i < d_; ++i) w[i] = a[i] * (s % m) % m;
    return w;
}

uint32_t WkArith::val(const Wel& a) const {
    uint32_t v = K_;
    for (uint32_t i = 0; i < d_; ++i) {
        uint64_t c = a[i];
        if (!c) continue;
        uint32_t e = 0;
        while (c % ell_ == 0) { c /= ell_; ++e; }
        v = std::min(v, e);
    }
    return v;
}

Wel WkArith::div_exact(const Wel& a, uint32_t v) const {
    Wel w{};
    for (uint32_t i = 0; i < d_; ++i) {
        ENGINE_CHECK_MSG(a[i] % pw_[v] == 0, "WkArith::div_exact: valuation below v");
        w[i] = a[i] / pw_[v];
    }
    return w;
}

Wel WkArith::div_floor(const Wel& a, uint32_t v) const {
    Wel w{};
    for (uint32_t i = 0; i < d_; ++i) w[i] = a[i] / pw_[v];
    return w;
}

Wel WkArith::inv(const Wel& u) const {
    const Fq::E u0 = reduce(u);
    ENGINE_CHECK_MSG(u0 != 0, "WkArith::inv: not a unit");
    Wel x = lift(F_->inv(u0));
    const Wel two = scalar(2);
    for (uint32_t it = 0; it < K_ + 1; ++it) x = mul(x, sub(two, mul(u, x)));   // Newton: precision doubles
    ENGINE_CHECK_MSG(mul(u, x) == one(), "WkArith::inv: the Newton lift failed");
    return x;
}

Wel WkArith::teichmuller(Fq::E x) const {
    ENGINE_CHECK_MSG(x != 0, "WkArith::teichmuller: x != 0");
    Wel y = lift(x);
    for (uint32_t i = 1; i < K_; ++i) {   // y^{q}, K - 1 times
        Wel r = one(), b = y;
        uint64_t e = F_->q();
        while (e) { if (e & 1) r = mul(r, b); b = mul(b, b); e >>= 1; }
        y = r;
    }
    return y;
}

std::string WkArith::text(const Wel& a) const {
    if (d_ == 1) return std::to_string(a[0]);
    std::ostringstream os;
    os << "[";
    for (uint32_t i = 0; i < d_; ++i) os << (i ? "," : "") << a[i];
    os << "]";
    return os.str();
}

std::string WkRouteName(WkRoute r) { return r == WkRoute::kDense ? "dense" : "sparse"; }

// ============================================================================
// Howell forms over the chain ring
// ============================================================================
namespace {

// r - q * t (both sorted); the entries that cancel are dropped.
WkRow MergeSub(const WkArith& W, const WkRow& r, const Wel& q, const WkRow& t, uint64_t& ops) {
    WkRow out;
    out.col.reserve(r.nnz() + t.nnz());
    out.val.reserve(r.nnz() + t.nnz());
    size_t i = 0, j = 0;
    while (i < r.nnz() || j < t.nnz()) {
        ++ops;
        if (j >= t.nnz() || (i < r.nnz() && r.col[i] < t.col[j])) {
            out.col.push_back(r.col[i]);
            out.val.push_back(r.val[i]);
            ++i;
        } else if (i >= r.nnz() || t.col[j] < r.col[i]) {
            const Wel v = W.sub(W.zero(), W.mul(q, t.val[j]));
            if (!W.is_zero(v)) { out.col.push_back(t.col[j]); out.val.push_back(v); }
            ++j;
        } else {
            const Wel v = W.sub(r.val[i], W.mul(q, t.val[j]));
            if (!W.is_zero(v)) { out.col.push_back(r.col[i]); out.val.push_back(v); }
            ++i;
            ++j;
        }
    }
    return out;
}

WkRow Scale(const WkArith& W, const WkRow& r, const Wel& s) {
    WkRow out;
    for (size_t k = 0; k < r.nnz(); ++k) {
        const Wel v = W.mul(s, r.val[k]);
        if (!W.is_zero(v)) { out.col.push_back(r.col[k]); out.val.push_back(v); }
    }
    return out;
}

WkRow ScaleInt(const WkArith& W, const WkRow& r, uint64_t s) {
    WkRow out;
    for (size_t k = 0; k < r.nnz(); ++k) {
        const Wel v = W.mul_scalar(r.val[k], s);
        if (!W.is_zero(v)) { out.col.push_back(r.col[k]); out.val.push_back(v); }
    }
    return out;
}

// Normalise r (leading valuation v) so that its leading entry is ell^v exactly.
void NormaliseLead(const WkArith& W, WkRow& r, uint32_t v) {
    const Wel u = W.div_exact(r.val[0], v);
    r = Scale(W, r, W.inv(u));
    ENGINE_CHECK_MSG(r.nnz() > 0 && r.val[0] == W.scalar(static_cast<int64_t>(W.ell_pow(v))),
                     "WkHowellForm: the normalised pivot is not ell^v");
}

// The canonical reduction: every entry above a pivot ell^v reduced coefficientwise into [0, ell^v).
// rows / piv / pval ascending by pivot.
void Canonicalise(const WkArith& W, uint32_t D, std::vector<WkRow>& rows, const std::vector<uint32_t>& piv,
                  const std::vector<uint32_t>& pval, uint64_t& ops) {
    std::vector<int32_t> row_of_col(D, -1);
    for (size_t i = 0; i < piv.size(); ++i) row_of_col[piv[i]] = static_cast<int32_t>(i);
    for (size_t ii = rows.size(); ii-- > 0;) {
        WkRow& r = rows[ii];
        size_t k = 1;
        while (k < r.nnz()) {
            const uint32_t c = r.col[k];
            const int32_t t = row_of_col[c];
            if (t >= 0) {
                const Wel q = W.div_floor(r.val[k], pval[static_cast<size_t>(t)]);
                if (!W.is_zero(q)) r = MergeSub(W, r, q, rows[static_cast<size_t>(t)], ops);
            }
            k = static_cast<size_t>(std::upper_bound(r.col.begin(), r.col.end(), c) - r.col.begin());
        }
    }
}

void DenseHowell(const WkArith& W, uint32_t D, std::vector<WkRow>& gens, std::vector<WkRow>& rows, std::vector<uint32_t>& piv,
                 std::vector<uint32_t>& pval, WkStats& st) {
    ENGINE_CHECK_MSG(D <= kWkDenseCap || D == 0, "WkHowellForm: dense route above kWkDenseCap");
    const uint32_t K = W.K();
    std::vector<std::vector<Wel>> pool;
    pool.reserve(gens.size() + 16);
    for (const WkRow& g : gens) {
        std::vector<Wel> v(D, W.zero());
        for (size_t k = 0; k < g.nnz(); ++k) v[g.col[k]] = g.val[k];
        pool.push_back(std::move(v));
    }
    gens.clear();
    std::vector<uint8_t> alive(pool.size(), 1);
    for (uint32_t c = 0; c < D; ++c) {
        int64_t best = -1;
        uint32_t bv = K;
        for (size_t r = 0; r < pool.size(); ++r) {
            if (!alive[r]) continue;
            const uint32_t v = W.val(pool[r][c]);
            if (v < bv) { bv = v; best = static_cast<int64_t>(r); if (bv == 0) break; }
        }
        if (best < 0) continue;
        std::vector<Wel> P = std::move(pool[static_cast<size_t>(best)]);
        alive[static_cast<size_t>(best)] = 0;
        const Wel ui = W.inv(W.div_exact(P[c], bv));
        for (uint32_t k = c; k < D; ++k) if (!W.is_zero(P[k])) P[k] = W.mul(ui, P[k]);
        ENGINE_CHECK_MSG(P[c] == W.scalar(static_cast<int64_t>(W.ell_pow(bv))), "WkHowellForm (dense): pivot not ell^v");
        for (size_t r = 0; r < pool.size(); ++r) {
            if (!alive[r] || W.is_zero(pool[r][c])) continue;
            const Wel q = W.div_exact(pool[r][c], bv);
            for (uint32_t k = c; k < D; ++k) {
                if (W.is_zero(P[k])) continue;
                pool[r][k] = W.sub(pool[r][k], W.mul(q, P[k]));
                ++st.merge_ops;
            }
            ENGINE_CHECK_MSG(W.is_zero(pool[r][c]), "WkHowellForm (dense): elimination left the pivot column");
        }
        if (bv > 0) {
            std::vector<Wel> A(D, W.zero());
            bool nz = false;
            for (uint32_t k = c + 1; k < D; ++k) {
                A[k] = W.mul_scalar(P[k], W.ell_pow(K - bv));
                nz = nz || !W.is_zero(A[k]);
            }
            if (nz) { pool.push_back(std::move(A)); alive.push_back(1); ++st.saturation_rows; }
        }
        WkRow out;
        for (uint32_t k = c; k < D; ++k) if (!W.is_zero(P[k])) { out.col.push_back(k); out.val.push_back(P[k]); }
        rows.push_back(std::move(out));
        piv.push_back(c);
        pval.push_back(bv);
    }
    for (size_t r = 0; r < pool.size(); ++r)
        if (alive[r])
            for (uint32_t k = 0; k < D; ++k)
                ENGINE_CHECK_MSG(W.is_zero(pool[r][k]), "WkHowellForm (dense): a non-zero row survived every column");
}

void SparseHowell(const WkArith& W, uint32_t D, std::vector<WkRow>& gens, std::vector<WkRow>& rows, std::vector<uint32_t>& piv,
                  std::vector<uint32_t>& pval, WkStats& st) {
    const uint32_t K = W.K();
    std::vector<int32_t> row_of_col(D, -1);
    std::vector<WkRow> B;
    std::vector<uint32_t> bval;
    std::vector<WkRow> work;
    for (size_t gi = 0; gi < gens.size(); ++gi) {
        work.push_back(std::move(gens[gi]));
        while (!work.empty()) {
            WkRow r = std::move(work.back());
            work.pop_back();
            while (r.nnz() > 0) {
                const uint32_t c = r.col[0];
                const uint32_t w = W.val(r.val[0]);
                const int32_t idx = row_of_col[c];
                if (idx >= 0 && w >= bval[static_cast<size_t>(idx)]) {
                    const Wel q = W.div_exact(r.val[0], bval[static_cast<size_t>(idx)]);
                    r = MergeSub(W, r, q, B[static_cast<size_t>(idx)], st.merge_ops);
                    ENGINE_CHECK_MSG(r.nnz() == 0 || r.col[0] > c, "WkHowellForm (sparse): the pivot column survived");
                    continue;
                }
                NormaliseLead(W, r, w);
                if (w > 0) {
                    WkRow s = ScaleInt(W, r, W.ell_pow(K - w));
                    if (s.nnz()) { work.push_back(std::move(s)); ++st.saturation_rows; }
                }
                if (idx < 0) {
                    row_of_col[c] = static_cast<int32_t>(B.size());
                    B.push_back(std::move(r));
                    bval.push_back(w);
                    break;
                }
                // a smaller valuation displaces the stored row, which is reduced by the new one and continues.
                WkRow old = std::move(B[static_cast<size_t>(idx)]);
                B[static_cast<size_t>(idx)] = std::move(r);
                bval[static_cast<size_t>(idx)] = w;
                const Wel q = W.div_exact(old.val[0], w);
                r = MergeSub(W, old, q, B[static_cast<size_t>(idx)], st.merge_ops);
                ENGINE_CHECK_MSG(r.nnz() == 0 || r.col[0] > c, "WkHowellForm (sparse): displacement left the pivot column");
                ++st.displaced;
            }
        }
    }
    gens.clear();
    for (uint32_t c = 0; c < D; ++c) {
        const int32_t idx = row_of_col[c];
        if (idx < 0) continue;
        rows.push_back(std::move(B[static_cast<size_t>(idx)]));
        piv.push_back(c);
        pval.push_back(bval[static_cast<size_t>(idx)]);
    }
}

}  // namespace

void WkHowellForm(const WkArith& W, uint32_t D, std::vector<WkRow> gens, WkRoute route, std::vector<WkRow>& rows,
                  std::vector<uint32_t>& piv, std::vector<uint32_t>& pval, WkStats* stats) {
    WkStats local;
    WkStats& st = stats ? *stats : local;
    rows.clear();
    piv.clear();
    pval.clear();
    for (const WkRow& g : gens) {
        ENGINE_CHECK_MSG(g.col.size() == g.val.size() && std::is_sorted(g.col.begin(), g.col.end()), "WkHowellForm: a malformed row");
        ENGINE_CHECK_MSG(g.col.empty() || g.col.back() < D, "WkHowellForm: a column beyond D");
    }
    if (route == WkRoute::kDense) DenseHowell(W, D, gens, rows, piv, pval, st);
    else SparseHowell(W, D, gens, rows, piv, pval, st);
    Canonicalise(W, D, rows, piv, pval, st.merge_ops);
    st.final_nnz = 0;
    for (const WkRow& r : rows) st.final_nnz += r.nnz();
}

std::vector<uint32_t> WkCokernelExponents(const WkArith& W, uint32_t D, const std::vector<WkRow>& rows,
                                          const std::vector<uint32_t>& piv, const std::vector<uint32_t>& pval) {
    const uint32_t K = W.K();
    // The unit-pivot rows split off: their pivot columns carry nothing else (canonical residues mod ell^0 = 0).
    std::vector<int32_t> colmap(D, -1);
    std::vector<uint8_t> unitcol(D, 0);
    for (size_t i = 0; i < piv.size(); ++i) if (pval[i] == 0) unitcol[piv[i]] = 1;
    uint32_t Dp = 0;
    for (uint32_t c = 0; c < D; ++c) if (!unitcol[c]) colmap[c] = static_cast<int32_t>(Dp++);
    std::vector<std::vector<Wel>> T;
    for (size_t i = 0; i < rows.size(); ++i) {
        if (pval[i] == 0) continue;
        std::vector<Wel> v(Dp, W.zero());
        for (size_t k = 0; k < rows[i].nnz(); ++k) {
            const uint32_t c = rows[i].col[k];
            ENGINE_CHECK_MSG(!unitcol[c], "WkCokernelExponents: an entry in a unit-pivot column (not canonical)");
            v[static_cast<size_t>(colmap[c])] = rows[i].val[k];
        }
        T.push_back(std::move(v));
    }
    // Smith by full least-valuation pivoting over the chain ring.
    std::vector<uint32_t> diag;
    const size_t R = T.size();
    for (size_t t = 0; t < std::min<size_t>(R, Dp); ++t) {
        size_t br = R, bc = Dp;
        uint32_t bv = K;
        for (size_t r = t; r < R && bv > 0; ++r)
            for (size_t c = t; c < Dp; ++c) {
                const uint32_t v = W.val(T[r][c]);
                if (v < bv) { bv = v; br = r; bc = c; if (bv == 0) break; }
            }
        if (br == R) break;
        std::swap(T[t], T[br]);
        for (auto& row : T) std::swap(row[t], row[bc]);
        const Wel ui = W.inv(W.div_exact(T[t][t], bv));
        for (size_t c = t; c < Dp; ++c) T[t][c] = W.mul(ui, T[t][c]);
        for (size_t r = 0; r < R; ++r) {
            if (r == t || W.is_zero(T[r][t])) continue;
            const Wel q = W.div_exact(T[r][t], bv);
            for (size_t c = t; c < Dp; ++c) T[r][c] = W.sub(T[r][c], W.mul(q, T[t][c]));
        }
        // column operations clear row t (column t is zero off the diagonal now)
        for (size_t c = t + 1; c < Dp; ++c) T[t][c] = W.zero();
        diag.push_back(bv);
    }
    std::vector<uint32_t> ex;
    for (uint32_t v : diag) if (v > 0) ex.push_back(v);
    for (size_t i = diag.size(); i < Dp; ++i) ex.push_back(K);
    std::sort(ex.begin(), ex.end());
    return ex;
}

// ============================================================================
// NebWkCokernel
// ============================================================================
namespace {

// (gamma . f)_j = t(chi(a_h)) rho_n(h) f_{j.gamma}, h = h_j(gamma), over W_K.
struct WkMonomial {
    uint32_t m = 0, d = 0;
    std::vector<uint32_t> perm;
    std::vector<Wel> scal;
    std::vector<uint32_t> blk;
    std::vector<std::vector<uint64_t>> blocks;   // rho_n(h) mod ell^K, d x d
};

WkMonomial MonomialOver(const level::Gamma0LevelN& lvl, const Mat22& gamma, uint32_t n, const WkArith& W,
                        const std::vector<Wel>& chi_t, level::RhoCache& rho) {
    WkMonomial M;
    M.m = lvl.m();
    M.d = n + 1;
    M.perm.assign(M.m + 1, 0);
    M.scal.assign(M.m + 1, W.zero());
    M.blk.assign(M.m + 1, 0);
    std::vector<level::Block> store;
    const uint64_t N = std::max<uint64_t>(lvl.N(), 1);
    for (uint32_t j = 1; j <= M.m; ++j) {
        const Mat22 x = lvl.rep(j) * gamma;
        const uint32_t jj = lvl.class_of(x);
        const Mat22 h = x * lvl.rep(jj).inv_unimodular();
        ENGINE_CHECK_MSG(lvl.contains(h), "NebWkCokernel: h_j(gamma) not in Gamma_0(N) (lem:inducedchar)");
        M.perm[j] = jj;
        int64_t a = h.a % static_cast<int64_t>(N);
        if (a < 0) a += static_cast<int64_t>(N);
        M.scal[j] = chi_t[static_cast<size_t>(a)];
        ENGINE_CHECK_MSG(!W.is_zero(M.scal[j]), "NebWkCokernel: chi at a non-unit a_h");
        M.blk[j] = rho.block_of(h, store);
    }
    M.blocks.reserve(store.size());
    for (const level::Block& b : store) M.blocks.emplace_back(b.dense.begin(), b.dense.end());
    return M;
}

std::vector<Wel> BlockMulW(const WkArith& W, const WkMonomial& M, uint32_t j, const std::vector<Wel>& v) {
    const uint32_t d = M.d;
    std::vector<Wel> w(d, W.zero());
    const std::vector<uint64_t>& B = M.blocks[M.blk[j]];
    for (uint32_t r = 0; r < d; ++r) {
        Wel acc = W.zero();
        for (uint32_t c = 0; c < d; ++c) {
            const uint64_t b = B[static_cast<size_t>(r) * d + c];
            if (b && !W.is_zero(v[c])) acc = W.add(acc, W.mul_scalar(v[c], b));
        }
        w[r] = W.mul(M.scal[j], acc);
    }
    return w;
}

// prop:neborbit (1) over W_K: (1/o) sum_{i<o} M^i e_{j0, r} per orbit and r, zero vectors dropped; certifies
// gamma^o = id block by block first.
std::vector<WkRow> OrbitGenerators(const WkArith& W, const WkMonomial& M, uint32_t order, uint32_t& orbits) {
    const uint32_t m = M.m, d = M.d;
    std::vector<uint32_t> inv_perm(m + 1, 0);
    for (uint32_t j = 1; j <= m; ++j) inv_perm[M.perm[j]] = j;
    for (uint32_t j = 1; j <= m; ++j) {
        uint32_t k = j;
        for (uint32_t i = 0; i < order; ++i) k = M.perm[k];
        ENGINE_CHECK_MSG(k == j, "NebWkCokernel: gamma^o does not fix the coset");
        for (uint32_t r = 0; r < d; ++r) {
            std::vector<Wel> v(d, W.zero());
            v[r] = W.one();
            uint32_t b = j;
            for (uint32_t i = 0; i < order; ++i) { b = inv_perm[b]; v = BlockMulW(W, M, b, v); }
            for (uint32_t c = 0; c < d; ++c)
                ENGINE_CHECK_MSG(v[c] == (c == r ? W.one() : W.zero()),
                                 "NebWkCokernel: gamma^o acts non-trivially (-I acts by chi(-1)(-1)^n != 1)");
        }
    }
    const Wel inv_o = W.inv(W.scalar(order));
    std::vector<uint8_t> seen(m + 1, 0);
    std::vector<WkRow> out;
    orbits = 0;
    for (uint32_t j0 = 1; j0 <= m; ++j0) {
        if (seen[j0]) continue;
        ++orbits;
        uint32_t k = j0;
        for (uint32_t i = 0; i < order; ++i) { seen[k] = 1; k = M.perm[k]; }
        for (uint32_t r = 0; r < d; ++r) {
            std::map<uint32_t, std::vector<Wel>> acc;
            std::vector<Wel> v(d, W.zero());
            v[r] = W.one();
            uint32_t b = j0;
            for (uint32_t i = 0; i < order; ++i) {
                std::vector<Wel>& a = acc.emplace(b, std::vector<Wel>(d, W.zero())).first->second;
                for (uint32_t c = 0; c < d; ++c) a[c] = W.add(a[c], W.mul(inv_o, v[c]));
                b = inv_perm[b];
                v = BlockMulW(W, M, b, v);
            }
            WkRow g;
            for (const auto& kv : acc)
                for (uint32_t c = 0; c < d; ++c)
                    if (!W.is_zero(kv.second[c])) {
                        g.col.push_back((kv.first - 1) * d + c);
                        g.val.push_back(kv.second[c]);
                    }
            if (g.nnz()) out.push_back(std::move(g));
        }
    }
    return out;
}

uint32_t FreeRankOf(const WkArith& W, uint32_t D, const std::vector<WkRow>& gens, WkRoute route) {
    std::vector<WkRow> rows;
    std::vector<uint32_t> piv, pval;
    WkHowellForm(W, D, gens, route, rows, piv, pval);
    uint64_t len = 0;
    for (uint32_t v : pval) len += W.K() - v;
    ENGINE_CHECK_MSG(len % W.K() == 0, "NebWkCokernel: M^<gamma> is not free (its length is not a multiple of K)");
    return static_cast<uint32_t>(len / W.K());
}

}  // namespace

NebWkCokernel::NebWkCokernel(const level::Gamma0LevelN& lvl, uint32_t n, const Fq& F, uint32_t K, const CharacterValues& chi,
                             WkRoute route, bool with_ranks)
    : W_(F, K), route_(route), with_ranks_(with_ranks) {
    if (F.ell() < 5)
        throw NebentypusRefusal("nebentypus_small_ell", "ell in {2, 3}: no cokernel and no comparison with group cohomology");
    ENGINE_CHECK_MSG(chi.N == std::max<uint64_t>(lvl.N(), 1), "NebWkCokernel: chi has another modulus");
    const uint64_t D64 = static_cast<uint64_t>(lvl.m()) * (static_cast<uint64_t>(n) + 1);
    ENGINE_CHECK_MSG(D64 < (1ull << 31), "NebWkCokernel: D < 2^31");
    if (route == WkRoute::kDense && D64 > kWkDenseCap)
        throw NebentypusRefusal("wk_dense_cap", "D = " + std::to_string(D64) + " > " + std::to_string(kWkDenseCap) + " (dense W_K route)");
    D_ = static_cast<uint32_t>(D64);
    const Fq::E chim1 = lvl.N() <= 2 ? 1u : chi.at(static_cast<int64_t>(lvl.N()) - 1);
    const Fq::E eps = (n % 2 == 0) ? chim1 : F.neg(chim1);
    if (eps != 1) {
        ENGINE_CHECK_MSG(eps == F.neg(1), "NebWkCokernel: chi(-1)^2 != 1");
        parity_zero_ = true;
        return;
    }
    const auto t0 = std::chrono::steady_clock::now();
    // t o chi as a table on Z/N (lem:teichmuller (3)); zero at non-units.
    const uint64_t N = std::max<uint64_t>(lvl.N(), 1);
    std::vector<Wel> chi_t(N, W_.zero());
    for (uint64_t a = 0; a < N; ++a) {
        const Fq::E x = N == 1 ? 1u : chi.val[a];   // 0 at a non-unit
        if (x == 0) continue;
        chi_t[a] = W_.teichmuller(x);
    }
    const engine::ModCtx<uint32_t> ctx = K == 1 ? engine::ModCtx<uint32_t>(static_cast<uint32_t>(F.ell()))
                                                : engine::ModCtx<uint32_t>::prime_power(static_cast<uint32_t>(F.ell()), K);
    level::RhoCache rho(ctx, n);
    const WkMonomial MS = MonomialOver(lvl, Mat22::S(), n, W_, chi_t, rho);
    level::RhoCache rho2(ctx, n);
    const WkMonomial MU = MonomialOver(lvl, Mat22::ST(), n, W_, chi_t, rho2);
    std::vector<WkRow> gS = OrbitGenerators(W_, MS, 2, orbits_S_);
    std::vector<WkRow> gU = OrbitGenerators(W_, MU, 3, orbits_U_);
    stats_.generators = gS.size() + gU.size();
    stats_.generators_s = Seconds(t0);
    if (with_ranks_) {
        rank_S_ = FreeRankOf(W_, D_, gS, route_);
        rank_U_ = FreeRankOf(W_, D_, gU, route_);
    }
    std::vector<WkRow> gens = std::move(gS);
    for (WkRow& r : gU) gens.push_back(std::move(r));
    const auto t1 = std::chrono::steady_clock::now();
    WkHowellForm(W_, D_, std::move(gens), route_, rows_, piv_, pval_, &stats_);
    stats_.eliminate_s = Seconds(t1);
    const auto t2 = std::chrono::steady_clock::now();
    h1_exp_ = WkCokernelExponents(W_, D_, rows_, piv_, pval_);
    stats_.smith_s = Seconds(t2);
}

uint64_t NebWkCokernel::rel_length() const {
    uint64_t len = 0;
    for (uint32_t v : pval_) len += W_.K() - v;
    return len;
}

int64_t NebWkCokernel::h0_length() const {
    ENGINE_CHECK_MSG(with_ranks_, "NebWkCokernel::h0_length: built without the ranks of M^S, M^ST");
    if (parity_zero_) return 0;
    return static_cast<int64_t>(W_.K()) * (rank_S_ + rank_U_) - static_cast<int64_t>(rel_length());
}

uint32_t NebWkCokernel::h1_free_rank() const {
    return static_cast<uint32_t>(std::count(h1_exp_.begin(), h1_exp_.end(), W_.K()));
}

std::string NebWkCokernel::HowellDefects() const {
    const WkArith& W = W_;
    std::vector<int32_t> row_of_col(D_, -1);
    for (size_t i = 0; i < piv_.size(); ++i) {
        if (i > 0 && piv_[i] <= piv_[i - 1]) return "pivots not strictly ascending at row " + std::to_string(i);
        row_of_col[piv_[i]] = static_cast<int32_t>(i);
        const WkRow& r = rows_[i];
        if (r.nnz() == 0 || r.col[0] != piv_[i]) return "row " + std::to_string(i) + " does not start at its pivot";
        if (r.val[0] != W.scalar(static_cast<int64_t>(W.ell_pow(pval_[i])))) return "row " + std::to_string(i) + ": pivot not ell^v";
    }
    for (size_t i = 0; i < rows_.size(); ++i)
        for (size_t k = 1; k < rows_[i].nnz(); ++k) {
            const int32_t t = row_of_col[rows_[i].col[k]];
            if (t < 0) continue;
            for (uint32_t e = 0; e < W.d(); ++e)
                if (rows_[i].val[k][e] >= W.ell_pow(pval_[static_cast<size_t>(t)]))
                    return "row " + std::to_string(i) + ": a non-canonical entry above pivot " + std::to_string(piv_[static_cast<size_t>(t)]);
        }
    uint64_t ops = 0;
    for (size_t i = 0; i < rows_.size(); ++i) {
        if (pval_[i] == 0) continue;
        WkRow s = ScaleInt(W, rows_[i], W.ell_pow(W.K() - pval_[i]));
        while (s.nnz()) {
            const int32_t t = row_of_col[s.col[0]];
            if (t < 0 || static_cast<size_t>(t) <= i || W.val(s.val[0]) < pval_[static_cast<size_t>(t)])
                return "row " + std::to_string(i) + ": the saturation ell^{K-v} row is not in the span of the rows below";
            const Wel q = W.div_exact(s.val[0], pval_[static_cast<size_t>(t)]);
            s = MergeSub(W, s, q, rows_[static_cast<size_t>(t)], ops);
        }
    }
    return "";
}

} // namespace neb
} // namespace hecke
