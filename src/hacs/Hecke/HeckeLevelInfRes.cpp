// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// (hecke-extensions slot 23, LES-3) Route B: the tree complex of a level pair -- see HeckeLevelInfRes.h.
#include "HeckeLevelInfRes.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <map>
#include <sstream>
#include <unordered_map>
#include <utility>

#include "../../ModArithmetic/EngineCheck.h"
#include "../Util/Timing.h"
#include "FpRank.h"

namespace hecke {
namespace level {
namespace les {

namespace {

using oc_util::Seconds;

// The rank mod ell of a set of dense rows (each of length ncols); destroys its input.
uint32_t RankRowsModEll(const std::vector<std::vector<uint32_t>>& rows, uint32_t ncols, uint32_t ell) {
    std::vector<uint32_t> flat;
    flat.reserve(rows.size() * ncols);
    for (const auto& r : rows) flat.insert(flat.end(), r.begin(), r.end());
    return RankModPrime(flat.data(), static_cast<uint32_t>(rows.size()), ncols, ell);
}

// Column access to a FibreMatrix: per source block column J (1-based), the (row block, block) pairs.
struct ColumnAccess {
    std::vector<std::vector<std::pair<uint32_t, const std::vector<uint32_t>*>>> col;
    explicit ColumnAccess(const FibreMatrix& F) : col(static_cast<size_t>(F.m_src()) + 1) {
        for (uint32_t i = 1; i <= F.m_dst(); ++i)
            for (const auto& e : F.row(i)) col[e.first].push_back(std::make_pair(i, &e.second));
    }
};

// Sparse accumulation over Z/Q.
struct Accum {
    uint64_t Q;
    std::map<uint32_t, uint64_t> m;
    explicit Accum(uint64_t q) : Q(q) {}
    void add(uint32_t c, uint64_t v) {
        if (v % Q == 0) return;
        uint64_t& x = m[c];
        x = (x + v) % Q;
    }
    SparseVec take() const {
        SparseVec s;
        for (const auto& kv : m)
            if (kv.second) { s.col.push_back(kv.first); s.val.push_back(static_cast<uint32_t>(kv.second)); }
        return s;
    }
};

} // namespace

engine::ModCtx<uint32_t> RingCtx(uint32_t ell, uint32_t K) {
    ENGINE_CHECK_MSG(K >= 1, "RingCtx: K >= 1");
    if (K == 1) return engine::ModCtx<uint32_t>(ell);
    return engine::ModCtx<uint32_t>::prime_power(ell, K);
}

// ---- TreeCokernel -----------------------------------------------------------------------------------------------------
TreeCokernel::TreeCokernel(const CosetLevel& A, const CharacterDatum& chi, uint32_t n_sym, uint32_t ell, uint32_t K, bool keep_generators,
                           uint64_t byte_cap)
    : A_(&A), d_(n_sym + 1), ell_(ell), K_(K), R_(ell, K) {
    const auto t0 = std::chrono::steady_clock::now();
    ENGINE_CHECK_MSG(ell >= 5, "TreeCokernel: ell >= 5 (6 must be a unit: prop:bredon (2) with the norms of S, U)");
    ENGINE_CHECK_MSG(static_cast<uint64_t>(A.m()) * d_ <= (1ull << 31), "TreeCokernel: D = m (n + 1) exceeds the column domain");
    D_ = A.m() * d_;
    const engine::ModCtx<uint32_t> ctx = RingCtx(ell, K);
    const uint64_t Q = R_.Q;
    RhoBlocks blocks(ctx, n_sym);
    const Mat22 mI = Mat22::I().neg();
    struct Family { Mat22 g; uint32_t order; bool minus; uint32_t* dim; };
    Family fam[3] = {{Mat22::S(), 4, false, &st_.dim_S}, {Mat22::ST(), 6, false, &st_.dim_U}, {mI, 2, true, &st_.dim_minus}};
    std::vector<SparseVec> gens;
    for (Family& f : fam) {
        const FibreMatrix M = TwistedMonomialMatrix(A, chi, f.g, n_sym, ctx, &blocks);
        const uint32_t m = A.m();
        std::vector<uint32_t> perm(m + 1, 0), inv(m + 1, 0);
        std::vector<const std::vector<uint32_t>*> blk(m + 1, nullptr);
        for (uint32_t j = 1; j <= m; ++j) {
            const auto& row = M.row(j);
            ENGINE_CHECK_MSG(row.size() == 1, "TreeCokernel: M(gamma) is monomial with one non-zero block per row");
            perm[j] = row[0].first;
            blk[j] = &row[0].second;
        }
        for (uint32_t j = 1; j <= m; ++j) {
            ENGINE_CHECK_MSG(inv[perm[j]] == 0, "TreeCokernel: M(gamma) permutes the classes");
            inv[perm[j]] = j;
        }
        std::vector<uint8_t> seen(m + 1, 0);
        uint32_t dim = 0;
        for (uint32_t c0 = 1; c0 <= m; ++c0) {
            if (seen[c0]) continue;
            std::vector<uint32_t> cyc;
            for (uint32_t c = c0; !seen[c]; c = perm[c]) { seen[c] = 1; cyc.push_back(c); }
            std::sort(cyc.begin(), cyc.end());
            std::vector<SparseVec> local;
            for (uint32_t r = 0; r < d_; ++r) {
                // M e_{(c, .)} lives at the class inv[c] with the vector B_{inv[c]} v ((M f)_j = B_j f_{perm j}).
                Accum acc(Q);
                std::vector<uint64_t> v(d_, 0), w(d_, 0);
                v[r] = 1;
                uint32_t c = c0;
                const uint32_t steps = f.minus ? 2u : f.order;
                for (uint32_t k = 0; k < steps; ++k) {
                    if (!f.minus || k == 0) {
                        for (uint32_t a = 0; a < d_; ++a) acc.add((c - 1) * d_ + a, v[a]);
                    } else {
                        for (uint32_t a = 0; a < d_; ++a) acc.add((c - 1) * d_ + a, (Q - v[a] % Q) % Q);   // 1 - M(-I)
                    }
                    const uint32_t nx = inv[c];
                    const std::vector<uint32_t>& B = *blk[nx];
                    for (uint32_t a = 0; a < d_; ++a) {
                        uint64_t s = 0;
                        for (uint32_t b = 0; b < d_; ++b) s = (s + static_cast<uint64_t>(B[static_cast<size_t>(a) * d_ + b]) * v[b]) % Q;
                        w[a] = s;
                    }
                    std::swap(v, w);
                    c = nx;
                }
                if (!f.minus) {
                    // M(gamma)^order = 1 (gamma^order = I in SL_2(Z)): the walk is back at e_r of c0.
                    bool back = c == c0;
                    for (uint32_t a = 0; a < d_ && back; ++a) back = v[a] % Q == (a == r ? 1u : 0u);
                    ENGINE_CHECK_MSG(back, "TreeCokernel: M(gamma)^ord != 1 on a unit vector");
                }
                SparseVec s = acc.take();
                if (s.nnz()) local.push_back(std::move(s));
            }
            // the orbit-wise free rank (mod ell) of the family on this cycle
            if (!local.empty()) {
                std::map<uint32_t, uint32_t> pos;
                for (const SparseVec& s : local)
                    for (uint32_t cc : s.col) pos.emplace(cc, 0);
                uint32_t k = 0;
                for (auto& kv : pos) kv.second = k++;
                std::vector<std::vector<uint32_t>> rows(local.size(), std::vector<uint32_t>(k, 0));
                for (size_t i = 0; i < local.size(); ++i)
                    for (size_t t = 0; t < local[i].col.size(); ++t) rows[i][pos[local[i].col[t]]] = local[i].val[t];
                dim += RankRowsModEll(rows, k, ell);
            }
            for (SparseVec& s : local) gens.push_back(std::move(s));
        }
        *f.dim = dim;
    }
    st_.generators = gens.size();
    if (keep_generators) gens_ = gens;
    Q_.reset(new tower::ChainCokernel(D_, std::move(gens), R_, byte_cap));
    st_.rank_rel = D_ - Q_->h();
    ENGINE_CHECK_MSG(st_.dim_S + st_.dim_U + st_.dim_minus >= st_.rank_rel, "TreeCokernel: rank Rel' exceeds the sum of the three ranks");
    st_.seconds = Seconds(t0);
}

// ---- dense helpers ----------------------------------------------------------------------------------------------------
DMat Identity(uint32_t h) {
    DMat I(h, h);
    for (uint32_t i = 0; i < h; ++i) I.at(i, i) = 1;
    return I;
}

DMat MatMul(const tower::ZK& R, const DMat& A, const DMat& B) {
    ENGINE_CHECK_MSG(A.c == B.r, "MatMul: shapes");
    DMat C(A.r, B.c);
    if (A.r == 0 || B.c == 0 || A.c == 0) return C;
    const uint64_t Q = R.Q, q1 = Q - 1;
    const uint64_t limit = std::max<uint64_t>(1, UINT64_MAX / (q1 * q1 + 1) - 1);
    std::vector<uint64_t> acc(B.c);
    for (uint32_t i = 0; i < A.r; ++i) {
        std::fill(acc.begin(), acc.end(), 0);
        uint64_t cnt = 0;
        for (uint32_t k = 0; k < A.c; ++k) {
            const uint64_t x = A.at(i, k);
            if (!x) continue;
            const uint32_t* b = &B.a[static_cast<size_t>(k) * B.c];
            for (uint32_t j = 0; j < B.c; ++j) acc[j] += x * b[j];
            if (++cnt >= limit) { for (uint64_t& v : acc) v %= Q; cnt = 0; }
        }
        for (uint32_t j = 0; j < B.c; ++j) C.at(i, j) = static_cast<uint32_t>(acc[j] % Q);
    }
    return C;
}

DMat MatAdd(const tower::ZK& R, const DMat& A, const DMat& B) {
    ENGINE_CHECK_MSG(A.r == B.r && A.c == B.c, "MatAdd: shapes");
    DMat C(A.r, A.c);
    for (size_t i = 0; i < A.a.size(); ++i) C.a[i] = R.add(A.a[i], B.a[i]);
    return C;
}

DMat MatSub(const tower::ZK& R, const DMat& A, const DMat& B) {
    ENGINE_CHECK_MSG(A.r == B.r && A.c == B.c, "MatSub: shapes");
    DMat C(A.r, A.c);
    for (size_t i = 0; i < A.a.size(); ++i) C.a[i] = R.sub(A.a[i], B.a[i]);
    return C;
}

DMat ReduceMod(const DMat& A, uint32_t m) {
    DMat C(A.r, A.c);
    for (size_t i = 0; i < A.a.size(); ++i) C.a[i] = A.a[i] % m;
    return C;
}

uint64_t Differ(const DMat& A, const DMat& B) {
    if (A.r != B.r || A.c != B.c) return UINT64_MAX;
    uint64_t d = 0;
    for (size_t i = 0; i < A.a.size(); ++i) d += (A.a[i] != B.a[i]);
    return d;
}

bool IsZero(const DMat& A) {
    for (uint32_t x : A.a)
        if (x) return false;
    return true;
}

uint32_t RankModEll(const DMat& A, uint32_t ell) {
    if (A.r == 0 || A.c == 0) return 0;
    std::vector<std::vector<uint32_t>> rows(A.r);
    for (uint32_t i = 0; i < A.r; ++i) rows[i].assign(A.a.begin() + static_cast<size_t>(i) * A.c, A.a.begin() + static_cast<size_t>(i + 1) * A.c);
    return RankRowsModEll(rows, A.c, ell);
}

uint32_t SpanLength(const tower::ZK& R, const DMat& A) {
    if (A.r == 0 || A.c == 0) return 0;
    std::vector<std::vector<uint32_t>> rows(A.c, std::vector<uint32_t>(A.r, 0));
    for (uint32_t i = 0; i < A.r; ++i)
        for (uint32_t j = 0; j < A.c; ++j) rows[j][i] = A.at(i, j);
    return tower::HowellZK(R, std::move(rows), A.r).length();
}

std::vector<uint32_t> JordanTypeUnipotent(const DMat& u, uint32_t ell, std::vector<uint32_t>* ranks_out) {
    ENGINE_CHECK_MSG(u.r == u.c, "JordanTypeUnipotent: square");
    const uint32_t h = u.r;
    const tower::ZK F(ell, 1);
    const DMat t = MatSub(F, ReduceMod(u, ell), Identity(h));
    std::vector<uint32_t> ranks{h};
    DMat P = t;
    for (uint32_t k = 1; k <= h + 1; ++k) {
        const uint32_t rk = RankModEll(P, ell);
        ranks.push_back(rk);
        if (rk == 0) break;
        ENGINE_CHECK_MSG(k <= h, "JordanTypeUnipotent: u - 1 is not nilpotent");
        P = MatMul(F, P, t);
    }
    // blocks of size >= k: ranks[k-1] - ranks[k]
    std::vector<uint32_t> ge(ranks.size() + 1, 0), jt(ranks.size() + 1, 0);
    for (size_t k = 1; k < ranks.size(); ++k) ge[k] = ranks[k - 1] - ranks[k];
    for (size_t k = 1; k < ranks.size(); ++k) jt[k] = ge[k] - ge[k + 1];
    if (ranks_out) *ranks_out = ranks;
    return jt;
}

std::string JordanText(const std::vector<uint32_t>& jt) {
    std::ostringstream os;
    bool first = true;
    for (size_t k = jt.size(); k-- > 1;)
        if (jt[k]) { os << (first ? "" : "+") << "J_" << k << "^" << jt[k]; first = false; }
    if (first) os << "0";
    return os.str();
}

// ---- maps on H^1 ------------------------------------------------------------------------------------------------------
DMat OnH1(const FibreMatrix& F, const TreeCokernel& src, const TreeCokernel& dst) {
    ENGINE_CHECK_MSG(F.m_src() == src.level().m() && F.m_dst() == dst.level().m() && F.d() == src.d() && F.d() == dst.d(),
                     "OnH1: the fibre map's shape does not match the two levels");
    ENGINE_CHECK_MSG(src.ring().Q == dst.ring().Q && F.p() == src.ring().Q, "OnH1: one ring");
    const ColumnAccess ca(F);
    const uint32_t d = src.d();
    const std::vector<uint32_t>& fc = src.Q().free_columns();
    DMat X(dst.h1(), src.h1());
    std::vector<uint32_t> out(dst.h1());
    for (uint32_t f = 0; f < fc.size(); ++f) {
        const uint32_t J = fc[f] / d + 1, r = fc[f] % d;
        std::fill(out.begin(), out.end(), 0);
        for (const auto& e : ca.col[J]) {
            const std::vector<uint32_t>& B = *e.second;
            for (uint32_t a = 0; a < d; ++a) {
                const uint32_t v = B[static_cast<size_t>(a) * d + r];
                if (v) dst.Q().add_coords((e.first - 1) * d + a, v, out);
            }
        }
        for (uint32_t i = 0; i < dst.h1(); ++i) X.at(i, f) = out[i];
    }
    return X;
}

uint64_t Escapes(const FibreMatrix& F, const TreeCokernel& src, const TreeCokernel& dst) {
    ENGINE_CHECK_MSG(!src.generators().empty() || src.stats().generators == 0, "Escapes: the source's generators were not kept");
    const ColumnAccess ca(F);
    const uint32_t d = src.d();
    const uint64_t Q = dst.ring().Q;
    uint64_t esc = 0;
    for (const SparseVec& g : src.generators()) {
        Accum acc(Q);
        for (size_t k = 0; k < g.col.size(); ++k) {
            const uint32_t J = g.col[k] / d + 1, r = g.col[k] % d;
            for (const auto& e : ca.col[J]) {
                const std::vector<uint32_t>& B = *e.second;
                for (uint32_t a = 0; a < d; ++a) {
                    const uint64_t v = B[static_cast<size_t>(a) * d + r];
                    if (v) acc.add((e.first - 1) * d + a, v * g.val[k] % Q);
                }
            }
        }
        if (!dst.Q().in_rel(acc.take())) ++esc;
    }
    return esc;
}

DMat OnH1(const SymbolOperator& X, const TreeCokernel& Q) {
    ENGINE_CHECK_MSG(X.D() == Q.D() && X.p() == Q.ring().Q, "OnH1(SymbolOperator): shape / ring");
    const std::vector<uint32_t>& fc = Q.Q().free_columns();
    DMat M(Q.h1(), Q.h1());
    std::vector<uint32_t> out(Q.h1());
    std::vector<std::pair<uint32_t, uint32_t>> img;
    for (uint32_t f = 0; f < fc.size(); ++f) {
        std::fill(out.begin(), out.end(), 0);
        img.clear();
        X.image(fc[f], img);
        for (const auto& e : img) Q.Q().add_coords(e.first, e.second, out);
        for (uint32_t i = 0; i < Q.h1(); ++i) M.at(i, f) = out[i];
    }
    return M;
}

uint64_t Escapes(const SymbolOperator& X, const TreeCokernel& Q) {
    ENGINE_CHECK_MSG(!Q.generators().empty() || Q.stats().generators == 0, "Escapes: the generators of Rel' were not kept (keep_generators)");
    const uint64_t P = Q.ring().Q;
    uint64_t esc = 0;
    std::vector<std::pair<uint32_t, uint32_t>> img;
    for (const SparseVec& g : Q.generators()) {
        Accum acc(P);
        for (size_t k = 0; k < g.col.size(); ++k) {
            img.clear();
            X.image(g.col[k], img);
            for (const auto& e : img) acc.add(e.first, static_cast<uint64_t>(e.second) * g.val[k] % P);
        }
        if (!Q.Q().in_rel(acc.take())) ++esc;
    }
    return esc;
}

DMat OnH1(const tower::ChiOperator& X, const TreeCokernel& Q) {
    ENGINE_CHECK_MSG(Q.d() == 1 && X.m() == Q.level().m(), "OnH1(ChiOperator): n = 0 and the same level");
    const std::vector<uint32_t>& fc = Q.Q().free_columns();
    DMat M(Q.h1(), Q.h1());
    std::vector<uint32_t> out(Q.h1());
    std::vector<std::pair<uint32_t, uint32_t>> img;
    for (uint32_t f = 0; f < fc.size(); ++f) {
        std::fill(out.begin(), out.end(), 0);
        img.clear();
        X.image(fc[f] + 1, img);
        for (const auto& e : img) Q.Q().add_coords(e.first - 1, e.second, out);
        for (uint32_t i = 0; i < Q.h1(); ++i) M.at(i, f) = out[i];
    }
    return M;
}

uint64_t Escapes(const tower::ChiOperator& X, const TreeCokernel& Q) {
    ENGINE_CHECK_MSG(!Q.generators().empty() || Q.stats().generators == 0, "Escapes: the generators of Rel' were not kept (keep_generators)");
    ENGINE_CHECK_MSG(Q.d() == 1, "Escapes(ChiOperator): n = 0");
    const uint64_t P = Q.ring().Q;
    uint64_t esc = 0;
    std::vector<std::pair<uint32_t, uint32_t>> img;
    for (const SparseVec& g : Q.generators()) {
        Accum acc(P);
        for (size_t k = 0; k < g.col.size(); ++k) {
            img.clear();
            X.image(g.col[k] + 1, img);
            for (const auto& e : img) acc.add(e.first - 1, static_cast<uint64_t>(e.second) * g.val[k] % P);
        }
        if (!Q.Q().in_rel(acc.take())) ++esc;
    }
    return esc;
}

RouteBSix RouteBSixTerm(const CosetLevel& A, const CharacterDatum& chiA, const CosetLevel& B, const CharacterDatum& chiB, uint32_t n_sym,
                        uint32_t ell) {
    const engine::ModCtx<uint32_t> ctx(ell);
    const RestrictionFibreMap R(A, chiA, B, chiB, n_sym, ctx);
    const TreeCokernel QA(A, chiA, n_sym, ell, 1, true), QB(B, chiB, n_sym, ell, 1, false);
    RouteBSix s;
    s.h0A = QA.h0(); s.h1A = QA.h1(); s.h0B = QB.h0(); s.h1B = QB.h1();
    s.rank_res1 = RankModEll(OnH1(R.alpha(), QA, QB), ell);
    s.H1c = s.h0B - s.h0A + s.ker_res1();
    s.H2c = s.h1B - s.rank_res1;
    s.alpha_escapes = Escapes(R.alpha(), QA, QB);
    return s;
}

// ---- oracle O3 ----------------------------------------------------------------------------------------------------------
uint32_t FiniteGroupH1SL2(uint32_t ell, uint32_t n_sym, uint32_t* VG_out, uint64_t* order_out) {
    ENGINE_CHECK_MSG(ell >= 2 && ell < 256, "FiniteGroupH1SL2: a small prime");
    const uint32_t D = n_sym + 1, p = ell;
    using E = std::array<uint32_t, 4>;   // (a, b, c, d) mod ell
    auto mul = [p](const E& x, const E& y) {
        return E{(x[0] * y[0] + x[1] * y[2]) % p, (x[0] * y[1] + x[1] * y[3]) % p, (x[2] * y[0] + x[3] * y[2]) % p, (x[2] * y[1] + x[3] * y[3]) % p};
    };
    auto key = [p](const E& x) { return ((x[0] * p + x[1]) * p + x[2]) * p + x[3]; };
    // binomials mod p
    std::vector<std::vector<uint64_t>> C(D + 1, std::vector<uint64_t>(D + 1, 0));
    for (uint32_t i = 0; i <= D; ++i) {
        C[i][0] = 1 % p;
        for (uint32_t j = 1; j <= i; ++j) C[i][j] = (C[i - 1][j - 1] + (j <= i - 1 ? C[i - 1][j] : 0)) % p;
    }
    auto pw = [p](uint64_t b, uint32_t e) { uint64_t r = 1 % p; b %= p; while (e--) r = r * b % p; return r; };
    // rho_n by substitution x -> a x + c y, y -> b x + d y on x^{n-k} y^k, images as columns (les_model.rho)
    auto rho = [&](const E& g) {
        const uint32_t n = n_sym;
        std::vector<uint64_t> m(static_cast<size_t>(D) * D, 0);
        for (uint32_t k = 0; k <= n; ++k)
            for (uint32_t s = 0; s <= n - k; ++s) {
                const uint64_t cs = C[n - k][s] * pw(g[0], n - k - s) % p * pw(g[2], s) % p;
                if (!cs) continue;
                for (uint32_t t = 0; t <= k; ++t) {
                    const uint64_t ct = C[k][t] * pw(g[1], k - t) % p * pw(g[3], t) % p;
                    m[static_cast<size_t>(s + t) * D + k] = (m[static_cast<size_t>(s + t) * D + k] + cs * ct) % p;
                }
            }
        return m;
    };
    const E I{1, 0, 0, 1}, S{0, p - 1, 1, 0}, T{1, 1, 0, 1};
    const std::vector<E> gens{S, T};
    const uint32_t nunk = static_cast<uint32_t>(gens.size()) * D;
    // f(g) as a D x nunk matrix of the unknowns (f(S), f(T))
    std::unordered_map<uint32_t, std::vector<uint64_t>> F;
    std::vector<E> order{I};
    F[key(I)] = std::vector<uint64_t>(static_cast<size_t>(D) * nunk, 0);
    std::vector<std::vector<uint32_t>> rows;
    for (size_t i = 0; i < order.size(); ++i) {
        const E g = order[i];
        const std::vector<uint64_t> Ag = rho(g);
        const std::vector<uint64_t> fg = F[key(g)];
        for (uint32_t si = 0; si < gens.size(); ++si) {
            const E h = mul(g, gens[si]);
            std::vector<uint64_t> val = fg;   // f(g s) = f(g) + g f(s)
            for (uint32_t a = 0; a < D; ++a)
                for (uint32_t b = 0; b < D; ++b)
                    val[static_cast<size_t>(a) * nunk + si * D + b] = (val[static_cast<size_t>(a) * nunk + si * D + b] + Ag[static_cast<size_t>(a) * D + b]) % p;
            auto it = F.find(key(h));
            if (it == F.end()) {
                F[key(h)] = val;
                order.push_back(h);
            } else {
                for (uint32_t a = 0; a < D; ++a) {
                    std::vector<uint32_t> r(nunk);
                    for (uint32_t u = 0; u < nunk; ++u)
                        r[u] = static_cast<uint32_t>((it->second[static_cast<size_t>(a) * nunk + u] + p - val[static_cast<size_t>(a) * nunk + u]) % p);
                    rows.push_back(std::move(r));
                }
            }
        }
    }
    const uint32_t Z = nunk - RankRowsModEll(rows, nunk, p);
    std::vector<std::vector<uint32_t>> inv_rows;
    for (const E& s : gens) {
        const std::vector<uint64_t> As = rho(s);
        for (uint32_t a = 0; a < D; ++a) {
            std::vector<uint32_t> r(D);
            for (uint32_t b = 0; b < D; ++b) r[b] = static_cast<uint32_t>((As[static_cast<size_t>(a) * D + b] + p - (a == b ? 1 : 0)) % p);
            inv_rows.push_back(std::move(r));
        }
    }
    const uint32_t VG = D - RankRowsModEll(inv_rows, D, p);
    const uint32_t B1 = D - VG;
    if (VG_out) *VG_out = VG;
    if (order_out) *order_out = order.size();
    ENGINE_CHECK_MSG(Z >= B1, "FiniteGroupH1SL2: Z^1 smaller than B^1");
    return Z - B1;
}

} // namespace les
} // namespace level
} // namespace hecke
