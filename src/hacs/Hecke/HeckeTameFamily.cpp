// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// (hecke-extensions slot 22, WAKE-1) The tame-family Hecke algebra on Gamma_H(N) over F_p -- see HeckeTameFamily.h and
// docs/implementation/slot22/WAKE1_spec.md.
#include "HeckeTameFamily.h"

#include <algorithm>
#include <chrono>
#include <stdexcept>

#include "CoefficientRing.h"   // hecke::IsPrimeU64
#include "DenseRref.h"
#include "HeckeLevelNSparse.h"
#include "../Util/Rss.h"
#include "../../Engine/ThreadPool.h"
#include "../../ModArithmetic/EngineCheck.h"
#include "../../ModArithmetic/ScalarMod.h"
#include "../Util/Timing.h"

namespace hecke {
namespace tame {

namespace {

using oc_util::Seconds;

uint32_t InvP(uint32_t a, uint32_t p) {
    uint64_t r = 1, b = a % p, e = p - 2;
    while (e) {
        if (e & 1) r = r * b % p;
        b = b * b % p;
        e >>= 1;
    }
    return static_cast<uint32_t>(r);
}

// Polynomial in a square matrix: sum c_i D^i.
FpMat PolyEval(const std::vector<uint32_t>& c, const FpMat& D, uint32_t p) {
    FpMat out(D.rows, D.cols), P = Identity(D.rows);
    for (uint32_t ci : c) {
        out = Add(out, Scale(P, ci, p), p);
        P = Mul(P, D, p);
    }
    return out;
}

} // namespace

// ============================================================================
// dense F_p
// ============================================================================
FpMat Identity(uint32_t n) {
    FpMat I(n, n);
    for (uint32_t i = 0; i < n; ++i) I.at(i, i) = 1;
    return I;
}

FpMat Mul(const FpMat& A, const FpMat& B, uint32_t p) {
    ENGINE_CHECK_MSG(A.cols == B.rows, "tame::Mul: shapes");
    ENGINE_CHECK_MSG(p < 65536, "tame::Mul: p < 2^16 (uint64 accumulation)");
    FpMat C(A.rows, B.cols);
    // (WAKE-3) Rows in blocks of 8 (one pass over B per block); uint32 accumulation when A.cols (p-1)^2 < 2^32 and every
    // entry is reduced (a sum of A.cols products then cannot overflow), uint64 otherwise; blocks spread over the engine's
    // thread pool for large products.  Every entry is the same sum reduced mod p: value-neutral.
    bool reduced_in = true;
    for (uint32_t x : A.a) reduced_in = reduced_in && x < p;
    for (uint32_t x : B.a) reduced_in = reduced_in && x < p;
    const bool narrow = reduced_in && static_cast<uint64_t>(A.cols) * (p - 1) * (p - 1) < (1ull << 32);
    const uint32_t kBlk = 8;
    const uint32_t nblk = (A.rows + kBlk - 1) / kBlk;
    auto run_block = [&](uint32_t blk) {
        const uint32_t i0 = blk * kBlk, i1 = std::min(A.rows, i0 + kBlk);
        if (narrow) {
            std::vector<uint32_t> acc(static_cast<size_t>(kBlk) * B.cols, 0);
            for (uint32_t k = 0; k < A.cols; ++k) {
                const uint32_t* brow = &B.a[static_cast<size_t>(k) * B.cols];
                for (uint32_t i = i0; i < i1; ++i) {
                    const uint32_t aik = A.at(i, k);
                    if (!aik) continue;
                    uint32_t* ac = &acc[static_cast<size_t>(i - i0) * B.cols];
                    for (uint32_t j = 0; j < B.cols; ++j) ac[j] += aik * brow[j];
                }
            }
            for (uint32_t i = i0; i < i1; ++i)
                for (uint32_t j = 0; j < B.cols; ++j) C.at(i, j) = acc[static_cast<size_t>(i - i0) * B.cols + j] % p;
        } else {
            std::vector<uint64_t> acc(B.cols);
            for (uint32_t i = i0; i < i1; ++i) {
                std::fill(acc.begin(), acc.end(), 0);
                for (uint32_t k = 0; k < A.cols; ++k) {
                    const uint64_t aik = A.at(i, k);
                    if (!aik) continue;
                    const uint32_t* brow = &B.a[static_cast<size_t>(k) * B.cols];
                    for (uint32_t j = 0; j < B.cols; ++j) acc[j] += aik * brow[j];
                }
                for (uint32_t j = 0; j < B.cols; ++j) C.at(i, j) = static_cast<uint32_t>(acc[j] % p);
            }
        }
    };
    const uint64_t work = static_cast<uint64_t>(A.rows) * A.cols * B.cols;
    if (work >= (1ull << 27) && nblk > 1) {
        engine::tape::ThreadPool& pool = engine::tape::ThreadPool::global();
        pool.parallel_for(nblk, pool.size(), [&](uint32_t t, uint32_t) { run_block(t); });
    } else {
        for (uint32_t b = 0; b < nblk; ++b) run_block(b);
    }
    return C;
}

FpMat Add(const FpMat& A, const FpMat& B, uint32_t p) {
    ENGINE_CHECK_MSG(A.rows == B.rows && A.cols == B.cols, "tame::Add: shapes");
    FpMat C(A.rows, A.cols);
    for (size_t i = 0; i < A.a.size(); ++i) C.a[i] = (A.a[i] + B.a[i]) % p;
    return C;
}

FpMat Sub(const FpMat& A, const FpMat& B, uint32_t p) {
    ENGINE_CHECK_MSG(A.rows == B.rows && A.cols == B.cols, "tame::Sub: shapes");
    FpMat C(A.rows, A.cols);
    for (size_t i = 0; i < A.a.size(); ++i) C.a[i] = (A.a[i] + p - B.a[i] % p) % p;
    return C;
}

FpMat Scale(const FpMat& A, uint64_t s, uint32_t p) {
    FpMat C(A.rows, A.cols);
    const uint64_t t = s % p;
    for (size_t i = 0; i < A.a.size(); ++i) C.a[i] = static_cast<uint32_t>(A.a[i] * t % p);
    return C;
}

FpMat Transpose(const FpMat& A) {
    FpMat T(A.cols, A.rows);
    for (uint32_t i = 0; i < A.rows; ++i)
        for (uint32_t j = 0; j < A.cols; ++j) T.at(j, i) = A.at(i, j);
    return T;
}

FpMat Power(const FpMat& A, uint64_t e, uint32_t p) {
    FpMat R = Identity(A.rows), B = A;
    while (e) {
        if (e & 1) R = Mul(R, B, p);
        e >>= 1;
        if (e) B = Mul(B, B, p);
    }
    return R;
}

bool IsZero(const FpMat& A) {
    return std::all_of(A.a.begin(), A.a.end(), [](uint32_t x) { return x == 0; });
}

bool Equal(const FpMat& A, const FpMat& B) { return A.rows == B.rows && A.cols == B.cols && A.a == B.a; }

// (S24 CL-G2, B3 (b)) the engine's RREF (DenseRref.h): leftmost pivots, leading 1, zero rows at the bottom -- unique, so
// the result (matrix and pivots) is the slot-16 Gauss--Jordan loop's (and the WAKE-3 lazy-reduction variant's) entrywise.
std::vector<uint32_t> Rref(FpMat& A, uint32_t p) {
    return RrefRowMajor<uint32_t>(A.a, A.rows, A.cols, p);
}

uint32_t Rank(FpMat A, uint32_t p) { return static_cast<uint32_t>(Rref(A, p).size()); }

FpMat NullSpace(const FpMat& A0, uint32_t p) {
    FpMat A = A0;
    const std::vector<uint32_t> piv = Rref(A, p);
    std::vector<int32_t> is_piv(A.cols, -1);
    for (size_t r = 0; r < piv.size(); ++r) is_piv[piv[r]] = static_cast<int32_t>(r);
    std::vector<uint32_t> fr;
    for (uint32_t c = 0; c < A.cols; ++c)
        if (is_piv[c] < 0) fr.push_back(c);
    FpMat K(A.cols, static_cast<uint32_t>(fr.size()));
    for (size_t k = 0; k < fr.size(); ++k) {
        K.at(fr[k], static_cast<uint32_t>(k)) = 1;
        for (size_t r = 0; r < piv.size(); ++r)
            K.at(piv[r], static_cast<uint32_t>(k)) = (p - A.at(static_cast<uint32_t>(r), fr[k])) % p;
    }
    return K;
}

FpMat SelectRows(const FpMat& A, const std::vector<uint32_t>& rows) {
    FpMat S(static_cast<uint32_t>(rows.size()), A.cols);
    for (size_t i = 0; i < rows.size(); ++i)
        for (uint32_t j = 0; j < A.cols; ++j) S.at(static_cast<uint32_t>(i), j) = A.at(rows[i], j);
    return S;
}

FpMat SolveSquare(const FpMat& A, const FpMat& B, uint32_t p) {
    ENGINE_CHECK_MSG(A.rows == A.cols && B.rows == A.rows, "tame::SolveSquare: shapes");
    const uint32_t n = A.rows;
    FpMat M(n, n + B.cols);
    for (uint32_t i = 0; i < n; ++i) {
        for (uint32_t j = 0; j < n; ++j) M.at(i, j) = A.at(i, j);
        for (uint32_t j = 0; j < B.cols; ++j) M.at(i, n + j) = B.at(i, j);
    }
    const std::vector<uint32_t> piv = Rref(M, p);
    ENGINE_CHECK_MSG(piv.size() >= n && (n == 0 || piv[n - 1] == n - 1), "tame::SolveSquare: singular");
    FpMat X(n, B.cols);
    for (uint32_t i = 0; i < n; ++i)
        for (uint32_t j = 0; j < B.cols; ++j) X.at(i, j) = M.at(i, n + j);
    return X;
}

std::vector<uint32_t> Charpoly(const FpMat& A0, uint32_t p) {
    // Hessenberg reduction by similarity, then the standard recurrence.
    ENGINE_CHECK_MSG(A0.rows == A0.cols, "tame::Charpoly: square");
    const uint32_t n = A0.rows;
    FpMat H = A0;
    for (uint32_t k = 0; k + 2 <= n; ++k) {
        uint32_t piv = k + 1;
        while (piv < n && H.at(piv, k) == 0) ++piv;
        if (piv == n) continue;
        if (piv != k + 1) {
            for (uint32_t j = 0; j < n; ++j) std::swap(H.at(piv, j), H.at(k + 1, j));
            for (uint32_t i = 0; i < n; ++i) std::swap(H.at(i, piv), H.at(i, k + 1));
        }
        const uint64_t iv = InvP(H.at(k + 1, k), p);
        for (uint32_t i = k + 2; i < n; ++i) {
            const uint64_t f = H.at(i, k) * iv % p;
            if (!f) continue;
            // row_i -= f row_{k+1}; col_{k+1} += f col_i
            for (uint32_t j = 0; j < n; ++j) H.at(i, j) = static_cast<uint32_t>((H.at(i, j) + (p - f) * H.at(k + 1, j)) % p);
            for (uint32_t r = 0; r < n; ++r) H.at(r, k + 1) = static_cast<uint32_t>((H.at(r, k + 1) + f * H.at(r, i)) % p);
        }
    }
    // P_0 = 1; P_k = (x - h_kk) P_{k-1} - sum_{i<k} h_{i,k} prod_{j=i+1..k} h_{j,j-1} P_{i-1}  (1-based)
    std::vector<std::vector<uint32_t>> P(n + 1);
    P[0] = {1};
    for (uint32_t k = 1; k <= n; ++k) {
        std::vector<uint64_t> poly(k + 1, 0);
        for (size_t t = 0; t < P[k - 1].size(); ++t) {
            poly[t + 1] = (poly[t + 1] + P[k - 1][t]) % p;
            poly[t] = (poly[t] + static_cast<uint64_t>(p - H.at(k - 1, k - 1)) * P[k - 1][t]) % p;
        }
        uint64_t prod = 1;
        for (uint32_t i = k - 1; i >= 1; --i) {
            prod = prod * H.at(i, i - 1) % p;   // h_{i+1, i} in 1-based
            const uint64_t coef = prod * H.at(i - 1, k - 1) % p;
            if (coef)
                for (size_t t = 0; t < P[i - 1].size(); ++t) poly[t] = (poly[t] + (p - coef) * P[i - 1][t]) % p;
        }
        P[k].assign(poly.begin(), poly.end());
    }
    return P[n];
}

bool SameColumnSpan(const FpMat& A, const FpMat& B, uint32_t p) {
    if (A.rows != B.rows) return false;
    FpMat AB(A.rows, A.cols + B.cols);
    for (uint32_t i = 0; i < A.rows; ++i) {
        for (uint32_t j = 0; j < A.cols; ++j) AB.at(i, j) = A.at(i, j);
        for (uint32_t j = 0; j < B.cols; ++j) AB.at(i, A.cols + j) = B.at(i, j);
    }
    const uint32_t r = Rank(AB, p);
    return r == Rank(A, p) && r == Rank(B, p);
}

// ============================================================================
// readings on Q
// ============================================================================
FpMat OnQuotient(const level::QuotientCoords& QC, const level::SymbolOperator& X) {
    ENGINE_CHECK_MSG(QC.D() == X.D() && QC.p() == X.p(), "tame::OnQuotient: operator and quotient differ");
    const uint32_t h = QC.h(), p = QC.p();
    FpMat T(h, h);
    std::vector<uint64_t> acc(h);
    std::vector<std::pair<uint32_t, uint32_t>> img;
    for (uint32_t f = 0; f < h; ++f) {
        std::fill(acc.begin(), acc.end(), 0);
        img.clear();
        X.image(QC.free_columns()[f], img);
        for (const auto& e : img)
            for (auto it = QC.begin(e.first); it != QC.end(e.first); ++it) acc[it->first] += static_cast<uint64_t>(e.second) * it->second;
        for (uint32_t i = 0; i < h; ++i) T.at(i, f) = static_cast<uint32_t>(acc[i] % p);
    }
    return T;
}

std::vector<uint32_t> PivotRows(const FpMat& W, uint32_t p) {
    FpMat Wt = Transpose(W);
    std::vector<uint32_t> piv = Rref(Wt, p);
    ENGINE_CHECK_MSG(piv.size() == W.cols, "tame::PivotRows: W is not of full column rank");
    return piv;
}

SubspaceReader::SubspaceReader(const level::QuotientCoords& QC, const FpMat& W) : QC_(&QC), W_(W) {
    ENGINE_CHECK_MSG(W.rows == QC.h(), "tame::SubspaceReader: shapes");
    const uint32_t p = QC.p(), w = W.cols, h = QC.h(), D = QC.D();
    sel_ = PivotRows(W, p);
    Wsel_inv_ = SolveSquare(SelectRows(W, sel_), Identity(w), p);
    std::vector<int32_t> selpos(h, -1);
    for (size_t s = 0; s < sel_.size(); ++s) selpos[sel_[s]] = static_cast<int32_t>(s);
    selrow_.assign(static_cast<size_t>(D) * w, 0);
    for (uint32_t c = 0; c < D; ++c)
        for (auto it = QC.begin(c); it != QC.end(c); ++it) {
            const int32_t s = selpos[it->first];
            if (s >= 0) selrow_[static_cast<size_t>(c) * w + static_cast<uint32_t>(s)] = it->second;
        }
}

FpMat SubspaceReader::read(const level::SymbolOperator& X) const {
    const level::QuotientCoords& QC = *QC_;
    ENGINE_CHECK_MSG(QC.D() == X.D() && QC.p() == X.p(), "tame::SubspaceReader::read: operator and quotient differ");
    const uint32_t p = QC.p(), w = W_.cols, h = QC.h();
    std::vector<uint64_t> TWsel(static_cast<size_t>(w) * w, 0);   // (s, i)
    std::vector<uint64_t> G(w);
    std::vector<std::pair<uint32_t, uint32_t>> img;
    for (uint32_t f = 0; f < h; ++f) {
        bool nz = false;
        for (uint32_t i = 0; i < w && !nz; ++i) nz = W_.at(f, i) != 0;
        if (!nz) continue;
        std::fill(G.begin(), G.end(), 0);
        img.clear();
        X.image(QC.free_columns()[f], img);
        for (const auto& e : img) {
            const uint32_t* row = &selrow_[static_cast<size_t>(e.first) * w];
            for (uint32_t s = 0; s < w; ++s) G[s] += static_cast<uint64_t>(e.second) * row[s];
        }
        for (uint32_t s = 0; s < w; ++s) {
            const uint64_t gs = G[s] % p;
            if (!gs) continue;
            for (uint32_t i = 0; i < w; ++i) TWsel[static_cast<size_t>(s) * w + i] += gs * W_.at(f, i);
        }
        if (f % 4096 == 4095)
            for (uint64_t& x : TWsel) x %= p;
    }
    FpMat B(w, w);
    for (size_t i = 0; i < TWsel.size(); ++i) B.a[i] = static_cast<uint32_t>(TWsel[i] % p);
    return Mul(Wsel_inv_, B, p);
}

FpMat RestrictFull(const FpMat& T, const FpMat& W, uint32_t p, bool* stable) {
    if (W.rows == W.cols && Equal(W, Identity(W.rows))) {
        if (stable) *stable = true;
        return T;
    }
    const FpMat TW = Mul(T, W, p);
    const std::vector<uint32_t> sel = PivotRows(W, p);
    const FpMat R = SolveSquare(SelectRows(W, sel), SelectRows(TW, sel), p);
    if (stable) *stable = Equal(Mul(W, R, p), TW);
    return R;
}

FpMat GenEigenspace(const FpMat& R, uint32_t a, uint32_t p) {
    FpMat P = Sub(R, Scale(Identity(R.rows), a, p), p);
    uint32_t r = Rank(P, p);
    while (r > 0) {
        FpMat P2 = Mul(P, P, p);
        const uint32_t r2 = Rank(P2, p);
        if (r2 == r) break;
        P = std::move(P2);
        r = r2;
    }
    return NullSpace(P, p);
}

FpMat BoundaryOnQuotient(const level::SparseCokernel& Q, const level::CuspData& C, bool* rel_ok) {
    const uint32_t p = Q.p(), h = Q.h1();
    ENGINE_CHECK_MSG(C.cusp.size() == static_cast<size_t>(Q.D()) + 1, "tame::BoundaryOnQuotient: n = 0 (one column per class)");
    auto delta_add = [&](uint32_t col, uint64_t coef, std::vector<uint64_t>& v) {
        const uint32_t J = col + 1;
        v[C.cusp[J]] += coef;
        v[C.cusp_of_S[J]] += (p - coef % p);
    };
    FpMat B(C.c, h);
    std::vector<uint64_t> v(C.c);
    for (uint32_t f = 0; f < h; ++f) {
        std::fill(v.begin(), v.end(), 0);
        delta_add(Q.free_columns()[f], 1, v);
        for (uint32_t i = 0; i < C.c; ++i) B.at(i, f) = static_cast<uint32_t>(v[i] % p);
    }
    bool ok = true;
    for (uint32_t c : Q.pivots()) {
        std::fill(v.begin(), v.end(), 0);
        delta_add(c, 1, v);
        const level::SparseVec& t = Q.tail_of(c);
        for (size_t i = 0; i < t.col.size(); ++i) delta_add(t.col[i], t.val[i], v);
        for (uint64_t x : v) ok = ok && x % p == 0;
    }
    if (rel_ok) *rel_ok = ok;
    return B;
}

// ============================================================================
// MatAlgebra
// ============================================================================
MatAlgebra::MatAlgebra(const std::vector<FpMat>& gens, uint32_t p) : p_(p) {
    ENGINE_CHECK_MSG(!gens.empty(), "MatAlgebra: no generators");
    w_ = gens[0].rows;
    for (const FpMat& g : gens) ENGINE_CHECK_MSG(g.rows == w_ && g.cols == w_, "MatAlgebra: generators of one square shape");
    const uint32_t L = w_ * w_;
    // RREF rows of the flattened basis, maintained fully reduced.
    std::vector<std::vector<uint32_t>> ech;
    std::vector<uint32_t> piv;
    auto insert = [&](const FpMat& X) -> bool {
        std::vector<uint64_t> v(X.a.begin(), X.a.end());
        for (size_t r = 0; r < ech.size(); ++r) {
            const uint64_t f = v[piv[r]] % p_;
            if (!f) continue;
            for (uint32_t j = 0; j < L; ++j)
                if (ech[r][j]) v[j] = (v[j] + (p_ - f) * ech[r][j]) % p_;
        }
        uint32_t c = 0;
        while (c < L && v[c] % p_ == 0) ++c;
        if (c == L) return false;
        const uint64_t iv = InvP(static_cast<uint32_t>(v[c] % p_), p_);
        std::vector<uint32_t> row(L);
        for (uint32_t j = 0; j < L; ++j) row[j] = static_cast<uint32_t>(v[j] % p_ * iv % p_);
        for (auto& e : ech) {
            const uint64_t f = e[c];
            if (!f) continue;
            for (uint32_t j = 0; j < L; ++j)
                if (row[j]) e[j] = static_cast<uint32_t>((e[j] + (p_ - f) * row[j]) % p_);
        }
        ech.push_back(std::move(row));
        piv.push_back(c);
        return true;
    };
    std::vector<FpMat> frontier{Identity(w_)};
    insert(frontier[0]);
    basis_.push_back(frontier[0]);
    while (!frontier.empty()) {
        std::vector<FpMat> next;
        for (const FpMat& X : frontier)
            for (const FpMat& g : gens) {
                FpMat Y = Mul(X, g, p_);
                if (insert(Y)) {
                    basis_.push_back(Y);
                    next.push_back(std::move(Y));
                }
            }
        frontier = std::move(next);
    }
    const uint32_t k = dim();
    flat_ = FpMat(k, L);
    for (uint32_t i = 0; i < k; ++i)
        for (uint32_t j = 0; j < L; ++j) flat_.at(i, j) = basis_[i].a[j];
    sel_ = piv;
    std::sort(sel_.begin(), sel_.end());
    // M[s][i] = flat_[i][sel_s]; coords c solve M c = x_sel.
    FpMat M(k, k);
    for (uint32_t s = 0; s < k; ++s)
        for (uint32_t i = 0; i < k; ++i) M.at(s, i) = flat_.at(i, sel_[s]);
    sel_inv_ = SolveSquare(M, Identity(k), p_);
}

std::vector<uint32_t> MatAlgebra::coords(const FpMat& X) const {
    ENGINE_CHECK_MSG(X.rows == w_ && X.cols == w_, "MatAlgebra::coords: shape");
    const uint32_t k = dim();
    FpMat xs(k, 1);
    for (uint32_t s = 0; s < k; ++s) xs.at(s, 0) = X.a[sel_[s]] % p_;
    const FpMat c = Mul(sel_inv_, xs, p_);
    std::vector<uint32_t> out(k);
    for (uint32_t i = 0; i < k; ++i) out[i] = c.at(i, 0);
    if (!Equal(from_coords(out), X)) throw std::logic_error("MatAlgebra::coords: the matrix is not in the algebra");
    return out;
}

bool MatAlgebra::contains(const FpMat& X) const {
    try {
        coords(X);
    } catch (const std::logic_error&) {
        return false;
    }
    return true;
}

FpMat MatAlgebra::from_coords(const std::vector<uint32_t>& c) const {
    FpMat X(w_, w_);
    std::vector<uint64_t> acc(static_cast<size_t>(w_) * w_, 0);
    for (size_t i = 0; i < c.size(); ++i) {
        if (!c[i]) continue;
        for (size_t j = 0; j < acc.size(); ++j) acc[j] += static_cast<uint64_t>(c[i]) * basis_[i].a[j];
    }
    for (size_t j = 0; j < acc.size(); ++j) X.a[j] = static_cast<uint32_t>(acc[j] % p_);
    return X;
}

const std::vector<std::vector<std::vector<uint32_t>>>& MatAlgebra::mult() const {
    if (mult_.empty()) {
        const uint32_t k = dim();
        mult_.assign(k, std::vector<std::vector<uint32_t>>(k));
        for (uint32_t i = 0; i < k; ++i)
            for (uint32_t j = 0; j < k; ++j) mult_[i][j] = coords(Mul(basis_[i], basis_[j], p_));
    }
    return mult_;
}

uint32_t MatAlgebra::quotient_dim(const std::vector<FpMat>& gens, FpMat* ideal_rref) const {
    const uint32_t k = dim();
    FpMat rows(static_cast<uint32_t>(gens.size()) * k, k);
    uint32_t r = 0;
    for (const FpMat& g : gens)
        for (uint32_t i = 0; i < k; ++i, ++r) {
            const std::vector<uint32_t> c = coords(Mul(g, basis_[i], p_));
            for (uint32_t j = 0; j < k; ++j) rows.at(r, j) = c[j];
        }
    const uint32_t rk = static_cast<uint32_t>(Rref(rows, p_).size());
    if (ideal_rref) {
        FpMat out(rk, k);
        for (uint32_t i = 0; i < rk; ++i)
            for (uint32_t j = 0; j < k; ++j) out.at(i, j) = rows.at(i, j);
        *ideal_rref = out;
    }
    return k - rk;
}

uint32_t MatAlgebra::radical_nilpotency() const {
    // m = span{ b - lambda(b) : b in the basis }, lambda(b) the unique eigenvalue of b in F_p (A local, split).
    std::vector<FpMat> m;
    for (const FpMat& b : basis_) {
        bool found = false;
        for (uint32_t lam = 0; lam < p_ && !found; ++lam) {
            const FpMat x = Sub(b, Scale(Identity(w_), lam, p_), p_);
            if (IsZero(Power(x, w_, p_))) {
                if (!IsZero(x)) m.push_back(x);
                found = true;
            }
        }
        if (!found) return 0;
    }
    // m^e: spans as flattened rows.
    auto span_rank = [&](const std::vector<FpMat>& xs) {
        if (xs.empty()) return 0u;
        FpMat F(static_cast<uint32_t>(xs.size()), w_ * w_);
        for (size_t i = 0; i < xs.size(); ++i)
            for (uint32_t j = 0; j < w_ * w_; ++j) F.at(static_cast<uint32_t>(i), j) = xs[i].a[j];
        return Rank(F, p_);
    };
    std::vector<FpMat> pw = m;
    for (uint32_t e = 1; e <= dim() + 1; ++e) {
        if (span_rank(pw) == 0) return e;
        std::vector<FpMat> nx;
        for (const FpMat& x : pw)
            for (const FpMat& y : m) {
                FpMat z = Mul(x, y, p_);
                if (!IsZero(z)) nx.push_back(std::move(z));
            }
        // keep a basis only
        if (!nx.empty()) {
            FpMat F(static_cast<uint32_t>(nx.size()), w_ * w_);
            for (size_t i = 0; i < nx.size(); ++i)
                for (uint32_t j = 0; j < w_ * w_; ++j) F.at(static_cast<uint32_t>(i), j) = nx[i].a[j];
            const uint32_t rk = static_cast<uint32_t>(Rref(F, p_).size());
            nx.clear();
            for (uint32_t i = 0; i < rk; ++i) {
                FpMat z(w_, w_);
                for (uint32_t j = 0; j < w_ * w_; ++j) z.a[j] = F.at(i, j);
                nx.push_back(std::move(z));
            }
        }
        pw = std::move(nx);
    }
    return 0;
}

// ============================================================================
// V2
// ============================================================================
const char* AnsatzName(ReducibilityAnsatz a) {
    switch (a) {
        case ReducibilityAnsatz::kStandard: return "standard";
        case ReducibilityAnsatz::kSwapped: return "swapped";
        case ReducibilityAnsatz::kTrivial: return "trivial";
        case ReducibilityAnsatz::kNoDiamond: return "nodiamond";
    }
    return "?";
}

ReducibilityResult ReducibilityIdeal(const MatAlgebra& A, const std::vector<FrobeniusDatum>& frob, uint32_t delta_order, bool wild,
                                     ReducibilityAnsatz ansatz) {
    const uint32_t p = A.p();
    ENGINE_CHECK_MSG(delta_order >= 1, "ReducibilityIdeal: |Delta| >= 1");
    ENGINE_CHECK_MSG(!wild || p == 5, "ReducibilityIdeal: the wild variant (the 5-part of (Z/125)^x) is written for p = 5 only");
    const uint32_t k = A.dim(), nu = wild ? 6 : 1, dl = delta_order, b = dl * nu;
    const uint32_t L = k * b, other = k * (b - 1);
    const auto& mult = A.mult();
    const std::vector<uint32_t> one = A.coords(Identity(A.w()));
    auto idx = [&](uint32_t s, uint32_t t) { return s * nu + t; };
    // natural index x * b + y  ->  position (the "(x) 1" block, y = 0, last)
    auto pos = [&](uint32_t nat) { const uint32_t x = nat / b, y = nat % b; return y == 0 ? other + x : x * (b - 1) + (y - 1); };
    auto binom = [&](uint64_t f, uint64_t t) -> uint64_t {
        uint64_t c = 1;
        for (uint64_t i = 0; i < t; ++i) c = c * (f - i) / (i + 1);
        return f < t ? 0 : c % p;
    };
    // X^e Y^f as (B index, coefficient) pairs; Y = 1 + u, Y^f = sum_{t < 6} C(f mod 25, t) u^t.
    auto mono = [&](int64_t e, int64_t f) {
        std::vector<std::pair<uint32_t, uint64_t>> out;
        const uint32_t s = static_cast<uint32_t>(((e % dl) + dl) % dl);
        if (!wild) { out.push_back({idx(s, 0), 1}); return out; }
        const uint64_t ff = static_cast<uint64_t>(((f % 25) + 25) % 25);
        for (uint32_t t = 0; t < nu; ++t) {
            const uint64_t c = binom(ff, t);
            if (c) out.push_back({idx(s, t), c});
        }
        return out;
    };
    // the RREF of JC over the positions, fully reduced
    std::vector<std::vector<uint32_t>> ech;
    std::vector<uint32_t> piv;
    auto reduce = [&](std::vector<uint64_t>& v) {
        for (size_t r = 0; r < ech.size(); ++r) {
            const uint64_t f = v[piv[r]] % p;
            if (!f) continue;
            const std::vector<uint32_t>& row = ech[r];
            for (uint32_t j = 0; j < L; ++j)
                if (row[j]) v[j] = (v[j] + (p - f) * row[j]) % p;
        }
    };
    auto insert = [&](const std::vector<uint64_t>& nat) -> bool {
        std::vector<uint64_t> v(L, 0);
        for (uint32_t i = 0; i < L; ++i) v[pos(i)] = nat[i] % p;
        reduce(v);
        uint32_t c = 0;
        while (c < L && v[c] % p == 0) ++c;
        if (c == L) return false;
        const uint64_t iv = InvP(static_cast<uint32_t>(v[c] % p), p);
        std::vector<uint32_t> row(L);
        for (uint32_t j = 0; j < L; ++j) row[j] = static_cast<uint32_t>(v[j] % p * iv % p);
        for (auto& e : ech) {
            const uint64_t f = e[c];
            if (!f) continue;
            for (uint32_t j = 0; j < L; ++j)
                if (row[j]) e[j] = static_cast<uint32_t>((e[j] + (p - f) * row[j]) % p);
        }
        ech.push_back(std::move(row));
        piv.push_back(c);
        return true;
    };
    // the multipliers: every basis element of A (L_i: x -> sum_x' mult[x][i][x']), X, and u (wild)
    auto mul_A = [&](const std::vector<uint64_t>& v, uint32_t i) {
        std::vector<uint64_t> out(L, 0);
        for (uint32_t x = 0; x < k; ++x)
            for (uint32_t y = 0; y < b; ++y) {
                const uint64_t a = v[x * b + y] % p;
                if (!a) continue;
                const std::vector<uint32_t>& m = mult[x][i];
                for (uint32_t x2 = 0; x2 < k; ++x2)
                    if (m[x2]) out[x2 * b + y] += a * m[x2];
            }
        for (uint64_t& z : out) z %= p;
        return out;
    };
    auto mul_X = [&](const std::vector<uint64_t>& v) {
        std::vector<uint64_t> out(L, 0);
        for (uint32_t x = 0; x < k; ++x)
            for (uint32_t s = 0; s < dl; ++s)
                for (uint32_t t = 0; t < nu; ++t) out[x * b + idx((s + 1) % dl, t)] = v[x * b + idx(s, t)];
        return out;
    };
    auto mul_u = [&](const std::vector<uint64_t>& v) {
        std::vector<uint64_t> out(L, 0);
        for (uint32_t x = 0; x < k; ++x)
            for (uint32_t s = 0; s < dl; ++s)
                for (uint32_t t = 0; t + 1 < nu; ++t) out[x * b + idx(s, t + 1)] = v[x * b + idx(s, t)];
        return out;
    };
    // the relations E_q
    std::vector<std::vector<uint64_t>> queue;
    for (const FrobeniusDatum& fr : frob) {
        std::vector<uint64_t> E(L, 0);
        const uint64_t q = fr.q % p;
        auto add = [&](const std::vector<uint32_t>& a, uint64_t scal, const std::vector<std::pair<uint32_t, uint64_t>>& bm) {
            for (uint32_t x = 0; x < k; ++x) {
                const uint64_t ax = a[x] * scal % p;
                if (!ax) continue;
                for (const auto& yc : bm) E[x * b + yc.first] = (E[x * b + yc.first] + (p - ax * yc.second % p)) % p;
            }
        };
        for (uint32_t x = 0; x < k; ++x) E[x * b + idx(0, 0)] = (E[x * b + idx(0, 0)] + fr.Tq[x]) % p;
        const int64_t e = fr.e, f = fr.f;
        switch (ansatz) {
            case ReducibilityAnsatz::kStandard:
                add(one, q, mono(e, f));
                add(fr.Dq, 1, mono(-e, -f));
                break;
            case ReducibilityAnsatz::kSwapped:
                add(one, 1, mono(e, f));
                add(fr.Dq, q, mono(-e, -f));
                break;
            case ReducibilityAnsatz::kTrivial:
                add(one, q, mono(0, 0));
                add(fr.Dq, 1, mono(0, 0));
                break;
            case ReducibilityAnsatz::kNoDiamond:
                add(one, q, mono(e, f));
                add(one, 1, mono(-e, -f));
                break;
        }
        if (insert(E)) queue.push_back(std::move(E));
    }
    for (size_t qi = 0; qi < queue.size(); ++qi) {
        std::vector<std::vector<uint64_t>> cand;
        for (uint32_t i = 0; i < k; ++i) cand.push_back(mul_A(queue[qi], i));
        cand.push_back(mul_X(queue[qi]));
        if (wild) cand.push_back(mul_u(queue[qi]));
        for (auto& c : cand)
            if (insert(c)) queue.push_back(std::move(c));
    }
    ReducibilityResult R;
    R.relations = static_cast<uint32_t>(frob.size());
    // J_0 (x) 1 = JC cap (A (x) 1): the rows pivoting in the last block (zero on every other column).
    std::vector<std::vector<uint32_t>> j0;
    for (size_t r = 0; r < ech.size(); ++r)
        if (piv[r] >= other) j0.push_back(std::vector<uint32_t>(ech[r].begin() + other, ech[r].end()));
    std::sort(j0.begin(), j0.end(), [](const std::vector<uint32_t>& a, const std::vector<uint32_t>& c) {
        const auto lead = [](const std::vector<uint32_t>& v) { size_t i = 0; while (i < v.size() && !v[i]) ++i; return i; };
        return lead(a) < lead(c);
    });
    R.J0 = FpMat(static_cast<uint32_t>(j0.size()), k);
    for (size_t r = 0; r < j0.size(); ++r)
        for (uint32_t x = 0; x < k; ++x) R.J0.at(static_cast<uint32_t>(r), x) = j0[r][x];
    R.dim_quotient = k - static_cast<uint32_t>(j0.size());
    // surjectivity: 1 (x) X (and 1 (x) Y) in A (x) 1 + JC iff the normal form vanishes off the last block.
    auto in_A1_plus_JC = [&](const std::vector<uint64_t>& nat) {
        std::vector<uint64_t> v(L, 0);
        for (uint32_t i = 0; i < L; ++i) v[pos(i)] = nat[i] % p;
        reduce(v);
        for (uint32_t j = 0; j < other; ++j)
            if (v[j] % p) return false;
        return true;
    };
    std::vector<uint64_t> oneX(L, 0);
    for (uint32_t x = 0; x < k; ++x) oneX[x * b + idx(1 % dl, 0)] = one[x];
    R.surjective = in_A1_plus_JC(oneX);
    if (wild) {
        std::vector<uint64_t> oneY(L, 0);
        for (uint32_t x = 0; x < k; ++x) { oneY[x * b + idx(0, 0)] = one[x]; oneY[x * b + idx(0, 1)] = one[x]; }
        R.surjective = R.surjective && in_A1_plus_JC(oneY);
    }
    return R;
}

// ============================================================================
// the cell drivers
// ============================================================================
namespace {

std::vector<uint64_t> PrimeFactors(uint64_t N) {
    std::vector<uint64_t> out;
    for (uint64_t q = 2; q * q <= N; ++q)
        if (N % q == 0) {
            out.push_back(q);
            while (N % q == 0) N /= q;
        }
    if (N > 1) out.push_back(N);
    return out;
}

// The space Q of one level with its cusp data and boundary; operators memoised by label.
struct Space {
    const level::GammaHLevelN* G = nullptr;
    uint32_t p = 0;
    engine::ModCtx<uint32_t> ctx;
    std::unique_ptr<level::SparseCokernel> Q;
    std::unique_ptr<level::QuotientCoords> QC;
    level::CuspData C;
    FpMat delta;
    bool rel_ok = false;
    uint32_t delta_rank = 0;
    uint32_t escapes = 0;
    std::map<uint64_t, FpMat> T;      // full matrices of T_n / U_n on Q
    std::map<uint64_t, FpMat> Dia;    // full matrices of <d> on Q

    Space(const level::GammaHLevelN& lvl, uint32_t pp) : G(&lvl), p(pp), ctx(pp) {
        Q.reset(new level::SparseCokernel(level::SparseCokernel::Build(lvl, 0, ctx, true)));
        QC.reset(new level::QuotientCoords(*Q));
        C = level::CuspsOf(lvl);
        delta = BoundaryOnQuotient(*Q, C, &rel_ok);
        delta_rank = Rank(delta, p);
    }
    uint32_t h() const { return Q->h1(); }
    const FpMat& Tn(uint64_t n, bool check_rel = true) {
        auto it = T.find(n);
        if (it != T.end()) return it->second;
        const level::GammaHHeilbronn X(*G, n, 0, ctx);
        // prop:heilbronn (1) on every kept generator: checked for n <= 13 (the localisation operators and the identities'
        // operators); for larger n the cost (generators x |Mer_n| x tail length) is not spent.
        if (check_rel && n <= 13) escapes += level::RelEscapesSym(*Q, X);
        return T[n] = OnQuotient(*QC, X);
    }
    const FpMat& Dd(uint64_t d, bool check_rel = true) {
        auto it = Dia.find(d);
        if (it != Dia.end()) return it->second;
        const level::GammaHDiamond X(*G, d, 0, ctx);
        if (check_rel) escapes += level::RelEscapesSym(*Q, X);
        return Dia[d] = OnQuotient(*QC, X);
    }
    // Localisation of the start basis W0 at the list (oracle's eis_ops order); `stable` false if a restriction failed.
    FpMat Localise(FpMat W, const std::vector<uint64_t>& loc_primes, bool* stable) {
        bool ok = true;
        std::vector<std::pair<uint64_t, uint32_t>> ops;
        for (uint64_t q : loc_primes)
            if (G->N() % q != 0 && q != p) ops.push_back({q, static_cast<uint32_t>((1 + q) % p)});
        for (uint64_t l : PrimeFactors(G->N())) ops.push_back({l, 1u});
        for (const auto& op : ops) {
            if (W.cols == 0) break;
            bool st = true;
            const FpMat R = RestrictFull(Tn(op.first), W, p, &st);
            ok = ok && st;
            const FpMat E = GenEigenspace(R, op.second, p);
            W = Mul(W, E, p);
        }
        if (stable) *stable = ok;
        return W;
    }
};

} // namespace

MPartReading RunMParts(const level::GammaHLevelN& G, uint32_t p, const std::vector<uint64_t>& loc_primes) {
    Space S(G, p);
    MPartReading r;
    r.m = G.m();
    r.h = S.h();
    r.c = S.C.c;
    r.twelve_g = S.C.twelve_g;
    r.genus_ok = S.C.genus_integral() && static_cast<int64_t>(r.h) == 2 * S.C.genus() + r.c - 1;
    r.rel_boundary_ok = S.rel_ok;
    r.boundary_rank = S.delta_rank;
    bool st1 = true, st2 = true;
    // The full m-part first; the cuspidal one is its intersection with ker(delta) (delta is Hecke-equivariant, so ker(delta) is
    // stable and the generalised eigenspaces of the restriction are the intersections).
    const FpMat Vf = S.Localise(Identity(S.h()), loc_primes, &st2);
    const FpMat Vc = Mul(Vf, NullSpace(Mul(S.delta, Vf, p), p), p);
    r.cusp_m = Vc.cols;
    r.full_m = Vf.cols;
    if (Vc.cols > 0 && G.N() % 3 != 0) {   // (WAKE-3) <3> needs 3 !| N (the survey's (5, 41, 3)); dim_T3 stays 0 there
        bool s = true;
        std::vector<FpMat> gens{RestrictFull(S.Tn(3), Vc, p, &s)};
        const FpMat D3 = RestrictFull(S.Dd(3), Vc, p, &s);
        if (G.index_H() > 1) gens.push_back(D3);
        r.dim_T3 = MatAlgebra(gens, p).dim();
        r.dim_D3_minus_1 = Rank(Sub(D3, Identity(D3.rows), p), p);
        st1 = st1 && s;
    }
    r.rel_escapes = S.escapes + (st1 && st2 ? 0u : 1000000u);   // an unstable restriction is reported as an escape
    return r;
}

namespace {

// (WAKE-3) The survey route (TameCellSpec::lean; WAKE3_spec.md section 1, item 3).
TameCellReading RunTameCellLean(const TameCellSpec& spec) {
    const auto t0 = std::chrono::steady_clock::now();
    const uint32_t p = spec.p;
    ENGINE_CHECK_MSG(hecke::IsPrimeU64(p) && p >= 5, "RunTameCell: p >= 5 prime");
    const uint64_t N = spec.l0 * spec.l1;
    const level::GammaHLevelN G = level::GammaHLevelN::PrimeToPart(N, p);
    TameCellReading R;
    R.N = N;
    R.H = G.H_text();
    R.m = G.m();
    const uint32_t dl = G.index_H();
    {
        uint64_t x = spec.g % N;
        uint32_t ord = 1;
        while (G.coset_rank(x) != 0) { x = x * spec.g % N; ++ord; }
        ENGINE_CHECK_MSG(ord == dl, "RunTameCell: g does not generate G_N / H");
    }
    Space S(G, p);
    R.h = S.h();
    R.c = S.C.c;
    R.nu2 = S.C.nu2;
    R.nu3 = S.C.nu3;
    R.twelve_g = S.C.twelve_g;
    R.genus_ok = S.C.genus_integral() && static_cast<int64_t>(R.h) == 2 * S.C.genus() + R.c - 1;
    R.rel_boundary_ok = S.rel_ok;
    R.boundary_rank = S.delta_rank;
    R.qc_nnz = S.QC->nnz();
    bool st = true;
    // the full m-part: each localisation operator dense on Q, built, restricted, freed (no memo)
    FpMat W;
    bool first = true;
    {
        std::vector<std::pair<uint64_t, uint32_t>> ops;
        for (uint64_t q : spec.loc_primes)
            if (N % q != 0 && q != p) ops.push_back({q, static_cast<uint32_t>((1 + q) % p)});
        for (uint64_t l : PrimeFactors(N)) ops.push_back({l, 1u});
        for (const auto& op : ops) {
            if (!first && W.cols == 0) break;
            FpMat T;
            {
                const level::GammaHHeilbronn X(G, op.first, 0, S.ctx);
                if (op.first <= 13) S.escapes += level::RelEscapesSym(*S.Q, X);
                T = OnQuotient(*S.QC, X);
            }
            if (first) {
                W = GenEigenspace(T, op.second, p);
                first = false;
            } else {
                bool s = true;
                const FpMat Rr = RestrictFull(T, W, p, &s);
                st = st && s;
                T = FpMat();
                W = Mul(W, GenEigenspace(Rr, op.second, p), p);
            }
        }
    }
    const FpMat& Vf = W;
    R.full_m = Vf.cols;
    const FpMat V = Vf.cols ? Mul(Vf, NullSpace(Mul(S.delta, Vf, p), p), p) : FpMat(S.h(), 0);
    R.cusp_m = V.cols;
    if (V.cols > 0 && spec.algebra) {
        const SubspaceReader rd(*S.QC, V);
        // T_g and <g> through the dense route with the stability check (the cheap pair), everything else on V
        bool s = true;
        FpMat Tg, D;
        {
            FpMat T = OnQuotient(*S.QC, level::GammaHHeilbronn(G, spec.g, 0, S.ctx));
            Tg = RestrictFull(T, V, p, &s);
            T = OnQuotient(*S.QC, level::GammaHDiamond(G, spec.g, 0, S.ctx));
            D = RestrictFull(T, V, p, &s);
        }
        st = st && s;
        std::vector<FpMat> gens{Tg, D};
        R.dimA_TgD = MatAlgebra(gens, p).dim();
        std::vector<FrobeniusDatum> frob;
        std::vector<FpMat> frobT, frobD;
        bool dpow = true;
        for (uint64_t q = 2; q < spec.v2_qmax; ++q) {
            if (!hecke::IsPrimeU64(q) || N % q == 0 || q == p) continue;
            FrobeniusDatum fd;
            fd.q = q;
            const uint32_t rq = G.coset_rank(q);
            uint64_t x = 1;
            uint32_t e = 0;
            while (G.coset_rank(x) != rq) { x = x * spec.g % N; ++e; }
            fd.e = e;
            frobT.push_back(q == spec.g ? Tg : rd.read(level::GammaHHeilbronn(G, q, 0, S.ctx)));
            frobD.push_back(rd.read(level::GammaHDiamond(G, q, 0, S.ctx)));
            dpow = dpow && Equal(frobD.back(), Power(D, e, p));
            gens.push_back(frobT.back());
            frob.push_back(std::move(fd));
        }
        R.diamonds_power = dpow;
        const FpMat Tp = rd.read(level::GammaHHeilbronn(G, p, 0, S.ctx));
        gens.push_back(Tp);
        for (uint64_t l : PrimeFactors(N)) gens.push_back(rd.read(level::GammaHHeilbronn(G, l, 0, S.ctx)));
        for (uint64_t u = 2; u < spec.members_umax; ++u)
            if (engine::scalar::gcd<uint64_t>(u, N) == 1) gens.push_back(rd.read(level::GammaHDiamond(G, u, 0, S.ctx)));
        // every generator commutes with T_g and <g> (a cheap consistency check of the subspace reads)
        bool comm = true;
        for (const FpMat& X : gens) comm = comm && Equal(Mul(X, Tg, p), Mul(Tg, X, p)) && Equal(Mul(X, D, p), Mul(D, X, p));
        R.identities["every operator read on V commutes with T_g and <g>"] = comm;
        const MatAlgebra A(gens, p);
        R.dimA = A.dim();
        R.members_inside = R.dimA_TgD == R.dimA;
        R.radical_nilpotency = A.radical_nilpotency();
        R.t5_unipotent = IsZero(Power(Sub(Tp, Identity(V.cols), p), V.cols, p));
        R.nonordinary_dim = GenEigenspace(Tp, 0, p).cols;
        if (spec.v2) {
            for (size_t i = 0; i < frob.size(); ++i) {
                frob[i].Tq = A.coords(frobT[i]);
                frob[i].Dq = A.coords(frobD[i]);
            }
            R.red = ReducibilityIdeal(A, frob, dl, false, ReducibilityAnsatz::kStandard);
        }
    }
    if (spec.full_algebra && Vf.cols > 0) {
        const SubspaceReader rf(*S.QC, Vf);
        std::vector<FpMat> gens;
        for (uint64_t q : spec.full_algebra_primes)
            if (N % q && q != p) gens.push_back(rf.read(level::GammaHHeilbronn(G, q, 0, S.ctx)));
        for (uint64_t l : PrimeFactors(N)) gens.push_back(rf.read(level::GammaHHeilbronn(G, l, 0, S.ctx)));
        gens.push_back(rf.read(level::GammaHDiamond(G, spec.g, 0, S.ctx)));
        R.full_alg_dim = MatAlgebra(gens, p).dim();
    }
    R.rel_escapes = S.escapes;
    if (!st) R.notes.push_back("a restriction to a localised subspace was not stable");
    R.rel_escapes += st ? 0u : 1000000u;
    R.seconds = Seconds(t0);
    R.peak_rss_mib = oc_util::PeakRssMiB();
    return R;
}

} // namespace

TameCellReading RunTameCell(const TameCellSpec& spec) {
    if (spec.lean) return RunTameCellLean(spec);
    const auto t0 = std::chrono::steady_clock::now();
    const uint32_t p = spec.p;
    ENGINE_CHECK_MSG(hecke::IsPrimeU64(p) && p >= 5, "RunTameCell: p >= 5 prime");
    const uint64_t N = spec.l0 * spec.l1;
    const level::GammaHLevelN G = level::GammaHLevelN::PrimeToPart(N, p);
    TameCellReading R;
    R.N = N;
    R.H = G.H_text();
    R.m = G.m();
    // Delta = G_N / H must be cyclic, generated by g.
    const uint32_t dl = G.index_H();
    {
        uint64_t x = spec.g % N;
        uint32_t ord = 1;
        while (G.coset_rank(x) != 0) { x = x * spec.g % N; ++ord; }
        ENGINE_CHECK_MSG(ord == dl, "RunTameCell: g does not generate G_N / H");
    }
    Space S(G, p);
    R.h = S.h();
    R.c = S.C.c;
    R.nu2 = S.C.nu2;
    R.nu3 = S.C.nu3;
    R.twelve_g = S.C.twelve_g;
    R.genus_ok = S.C.genus_integral() && static_cast<int64_t>(R.h) == 2 * S.C.genus() + R.c - 1;
    R.rel_boundary_ok = S.rel_ok;
    R.boundary_rank = S.delta_rank;
    bool st = true, s1 = true;
    // the full m-part, then the cuspidal one as its intersection with ker(delta) (as in RunMParts)
    const FpMat Vf = S.Localise(Identity(S.h()), spec.loc_primes, &s1);
    st = st && s1;
    R.full_m = Vf.cols;
    const FpMat V = Mul(Vf, NullSpace(Mul(S.delta, Vf, p), p), p);
    R.cusp_m = V.cols;
    // identities on Q (I6) and the inverse-normalisation control (N4)
    if (spec.identities) {
        const FpMat& T2 = S.Tn(2);
        const FpMat& T3 = S.Tn(3);
        const FpMat& Dg = S.Dd(spec.g);
        if (N % 3) R.identities["T3^2 - T9 = 3<3>"] = Equal(Sub(Mul(T3, T3, p), S.Tn(9), p), Scale(S.Dd(3), 3, p));
        if (N % 2 && N % 3) R.identities["T2 T3 = T6"] = Equal(Mul(T2, T3, p), S.Tn(6));
        if (N % 2 && N % 7) R.identities["T2 T7 = T14"] = Equal(Mul(T2, S.Tn(7), p), S.Tn(14));
        if (N % 2 && N % 3) R.identities["<2><3> = <6>"] = Equal(Mul(S.Dd(2), S.Dd(3), p), S.Dd(6));
        bool hid = true;
        for (uint64_t h : G.generators()) hid = hid && Equal(S.Dd(h), Identity(S.h()));
        R.identities["<h> = 1 (h in H)"] = hid;
        R.identities["<g>^|Delta| = 1"] = Equal(Power(Dg, dl, p), Identity(S.h()));
        std::vector<const FpMat*> ops{&T2, &T3, &S.Tn(7), &Dg, &S.Dd(2)};
        for (uint64_t l : PrimeFactors(N)) ops.push_back(&S.Tn(l));
        bool comm = true;
        for (size_t i = 0; i < ops.size(); ++i)
            for (size_t j = i + 1; j < ops.size(); ++j) comm = comm && Equal(Mul(*ops[i], *ops[j], p), Mul(*ops[j], *ops[i], p));
        R.identities["T_2, T_3, T_7, U_l, <g>, <2> commute"] = comm;
        if (N % 3) {
            const level::GammaHDiamond inv(G, 3, 0, S.ctx, true);
            const FpMat D3inv = OnQuotient(*S.QC, inv);
            R.identities["N4: T3^2 - T9 = 3<3>^inv (must FAIL)"] = Equal(Sub(Mul(T3, T3, p), S.Tn(9), p), Scale(D3inv, 3, p));
        }
    }
    if (V.cols > 0 && spec.algebra) {
        bool s = true;
        const FpMat Tg = RestrictFull(S.Tn(spec.g), V, p, &s);
        const FpMat D = RestrictFull(S.Dd(spec.g), V, p, &s);
        st = st && s;
        const MatAlgebra A({Tg, D}, p);
        R.dimA = A.dim();
        R.radical_nilpotency = A.radical_nilpotency();
        bool inside = true;
        for (uint64_t q = 2; q < spec.members_qmax; ++q) {
            if (!hecke::IsPrimeU64(q)) continue;
            const FpMat Tq = RestrictFull(S.Tn(q), V, p, &s);
            inside = inside && s && A.contains(Tq);
        }
        for (uint64_t u = 2; u < spec.members_umax; ++u) {
            if (engine::scalar::gcd<uint64_t>(u, N) != 1) continue;
            const FpMat Du = RestrictFull(S.Dd(u), V, p, &s);
            inside = inside && s && A.contains(Du);
        }
        R.members_inside = inside;
        // ordinary: T_p - 1 nilpotent on V; the generalised T_p = 0 eigenspace (N5)
        const FpMat Tp = RestrictFull(S.Tn(p), V, p, &s);
        R.t5_unipotent = IsZero(Power(Sub(Tp, Identity(V.cols), p), V.cols, p));
        R.nonordinary_dim = GenEigenspace(Tp, 0, p).cols;
        // the presentation and the decoys (W-P6b, W-P7, N3)
        const FpMat I = Identity(V.cols);
        const FpMat Z = Sub(D, I, p);
        FpMat Y;
        if (spec.pres) {
            const Presentation& pr = *spec.pres;
            const FpMat u = Power(PolyEval(pr.u_base, D, p), pr.u_power, p);
            Y = Add(Mul(Tg, u, p), PolyEval(pr.v, D, p), p);
            R.pres_done = true;
            R.z5 = IsZero(Power(Z, 5, p));
            R.rel = Equal(Mul(Y, Y, p), Scale(Mul(Z, Z, p), pr.mu, p));
            FpMat mons(10, V.cols * V.cols);
            FpMat Zi = I;
            for (uint32_t i = 0; i < 5; ++i) {
                const FpMat YZ = Mul(Y, Zi, p);
                for (uint32_t j = 0; j < V.cols * V.cols; ++j) { mons.at(2 * i, j) = Zi.a[j]; mons.at(2 * i + 1, j) = YZ.a[j]; }
                Zi = Mul(Zi, Z, p);
            }
            R.pres_rank = Rank(mons, p);
            R.dim_y = A.quotient_dim({Y});
            R.dim_ypz = A.quotient_dim({Add(Y, Z, p)});
            R.dim_ymz = A.quotient_dim({Sub(Y, Z, p)});
            const FpMat D2 = RestrictFull(S.Dd(2), V, p, &s);
            const FpMat Z2 = Sub(D2, I, p);
            R.shifted_z5 = IsZero(Power(Z2, 5, p));
            R.shifted_rel = Equal(Mul(Y, Y, p), Scale(Mul(Z2, Z2, p), pr.mu, p));
        }
        // V2
        if (spec.v2) {
            const SubspaceReader rd(*S.QC, V);
            std::vector<FrobeniusDatum> frob;
            bool dpow = true;
            for (uint64_t q = 2; q < spec.v2_qmax; ++q) {
                if (!hecke::IsPrimeU64(q) || N % q == 0 || q == p) continue;
                FrobeniusDatum fd;
                fd.q = q;
                const uint32_t rq = G.coset_rank(q);
                uint64_t x = 1;
                uint32_t e = 0;
                while (G.coset_rank(x) != rq) { x = x * spec.g % N; ++e; }
                fd.e = e;
                if (spec.wild) {
                    const uint64_t y = engine::scalar::powmod<uint64_t>(q, 76, 125);
                    uint32_t f = 0;
                    uint64_t z6 = 1;
                    while (z6 != y) { z6 = z6 * 6 % 125; ++f; ENGINE_CHECK_MSG(f < 25, "RunTameCell: q^76 not a power of 6 mod 125"); }
                    fd.f = f;
                }
                FpMat Tq;
                auto it = S.T.find(q);
                if (it != S.T.end()) Tq = RestrictFull(it->second, V, p, &s);
                else Tq = rd.read(level::GammaHHeilbronn(G, q, 0, S.ctx));
                const FpMat Dq = rd.read(level::GammaHDiamond(G, q, 0, S.ctx));
                dpow = dpow && Equal(Dq, Power(D, e, p));
                fd.Tq = A.coords(Tq);
                fd.Dq = A.coords(Dq);
                frob.push_back(std::move(fd));
            }
            R.diamonds_power = dpow;
            R.red = ReducibilityIdeal(A, frob, dl, false, ReducibilityAnsatz::kStandard);
            auto same = [&](const FpMat& J1, const FpMat& J2) {
                return SameColumnSpan(Transpose(J1), Transpose(J2), p);
            };
            if (spec.pres) {
                FpMat Iy, Ipz, Imz;
                A.quotient_dim({Y}, &Iy);
                A.quotient_dim({Add(Y, Z, p)}, &Ipz);
                A.quotient_dim({Sub(Y, Z, p)}, &Imz);
                R.J_is_y = same(R.red.J0, Iy);
                R.J_is_ypz = same(R.red.J0, Ipz);
                R.J_is_ymz = same(R.red.J0, Imz);
            }
            if (spec.wild && p == 5) {
                R.red_wild = ReducibilityIdeal(A, frob, dl, true, ReducibilityAnsatz::kStandard);
                R.wild_same = R.red_wild.dim_quotient == R.red.dim_quotient && same(R.red_wild.J0, R.red.J0);
            }
            if (spec.controls)
                for (ReducibilityAnsatz a : {ReducibilityAnsatz::kSwapped, ReducibilityAnsatz::kTrivial, ReducibilityAnsatz::kNoDiamond}) {
                    const ReducibilityResult rr = ReducibilityIdeal(A, frob, dl, false, a);
                    R.variants[AnsatzName(a)] = rr;
                    R.variant_same[AnsatzName(a)] = rr.dim_quotient == R.red.dim_quotient && same(rr.J0, R.red.J0);
                }
        }
        st = st && s;
    }
    if (spec.full && spec.full_algebra && Vf.cols > 0) {
        bool s = true;
        std::vector<FpMat> gens;
        for (uint64_t q : spec.full_algebra_primes)
            if (N % q && q != p) gens.push_back(RestrictFull(S.Tn(q), Vf, p, &s));
        for (uint64_t l : PrimeFactors(N)) gens.push_back(RestrictFull(S.Tn(l), Vf, p, &s));
        gens.push_back(RestrictFull(S.Dd(spec.g), Vf, p, &s));
        st = st && s;
        R.full_alg_dim = MatAlgebra(gens, p).dim();
    }
    R.rel_escapes = S.escapes;
    if (!st) R.notes.push_back("a restriction to a localised subspace was not stable");
    R.rel_escapes += st ? 0u : 1000000u;
    R.seconds = Seconds(t0);
    R.peak_rss_mib = oc_util::PeakRssMiB();
    return R;
}


// (slot 23, G-EVT23) the HPWWE presentations and the arm-(H) spec -- moved verbatim from HeckeTameFamilyValidation.cpp.
const Presentation* HpwwePresentation(uint64_t l1) {
    static const Presentation kPres23{{2, 3, 1}, 2, {4, 2, 1, 2, 2}, 2};   // u = (2 - 2D + D^2)^2
    static const Presentation kPres67{{4, 1, 3, 4, 4}, 1, {2, 3, 1, 2, 3}, 1};   // u = -1 + D + 3D^2 + 4D^3 + 4D^4
    return l1 == 23 ? &kPres23 : l1 == 67 ? &kPres67 : nullptr;
}

TameCellSpec HpwweSpec(uint64_t l1) {
    TameCellSpec spec;
    spec.l1 = l1;
    spec.pres = HpwwePresentation(l1);
    spec.wild = spec.pres != nullptr;
    return spec;
}

} // namespace tame
} // namespace hecke
