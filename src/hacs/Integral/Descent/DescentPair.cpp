// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// DESCENT milestone D4: the boundary pair and the cone over Z, glued across Z[1/6], Z_(2), Z_(3), at level 1 and Gamma_0(N).
// See the header; [C] = docs/latex/gl2_contract.tex (v1.23.0-pre.27, sec:descentpair and its code map), cited by label.
#include "DescentPair.h"
#include "DescentSmith.h"
#include "DescentParAsm.h"

#include <algorithm>
#include <chrono>
#if defined(__GLIBC__)
#include <malloc.h>   // malloc_trim
#endif
#include <map>
#include <memory>
#include <set>
#include <sstream>

#include "DescentLevelN.h"
#include "DescentTreePair.h"
#include "../Model/IntegralHecke.h"
#include "../../EquivariantCohomologyOnK.h"
#include "../../EquivariantMappingCone.h"
#include "../../Hecke/CoefficientRing.h"
#include "../../Hecke/HeckeLevelN.h"
#include "../../Hecke/HeckeSettings.h"
#include "../../Hecke/SL2ZBoundaryPair.h"
#include "../../LocalSystemSL2Z.h"
#include "../../LocalSystemSimplicialComplex.h"
#include "../../SimplicialCohomology.h"
#include "../../../ModArithmetic/DenseElim.h"
#include "../../../ModArithmetic/EngineCheck.h"
#include "../../Util/CancelPoint.h"   // (NBCONFIRM S5) cooperative cancellation points

namespace integral_descent {
namespace pair {

using engine::DenseBuf;
using U32Ctx = engine::ModCtx<uint32_t>;
using clk = std::chrono::steady_clock;
using hecke::Mat22;

namespace {

double since(clk::time_point t0) { return std::chrono::duration<double>(clk::now() - t0).count(); }

Mat Identity(uint32_t n) {
    Mat I(n, n);
    for (uint32_t i = 0; i < n; ++i) I.at(i, i) = 1;
    return I;
}
Mat Reduce(const Mat& A, const ZMod& Z) {
    Mat B = A;
    for (auto& x : B.a) x = Z.red(x);
    return B;
}
Mat Transpose(const Mat& A) {
    Mat B(A.c, A.r);
    for (uint32_t i = 0; i < A.r; ++i)
        for (uint32_t j = 0; j < A.c; ++j) B.at(j, i) = A.at(i, j);
    return B;
}
void Place(Mat& T, const Mat& B, uint32_t r0, uint32_t c0) {
    for (uint32_t i = 0; i < B.r; ++i) std::copy(B.row(i), B.row(i) + B.c, T.row(r0 + i) + c0);
}
// rows / columns selected by index lists
Mat Sub(const Mat& A, const std::vector<uint32_t>& rows, const std::vector<uint32_t>& cols) {
    Mat B(static_cast<uint32_t>(rows.size()), static_cast<uint32_t>(cols.size()));
    for (uint32_t i = 0; i < rows.size(); ++i) {
        const uint32_t* a = A.row(rows[i]);
        uint32_t* b = B.row(i);
        for (uint32_t j = 0; j < cols.size(); ++j) b[j] = a[cols[j]];
    }
    return B;
}
bool EqualMat(const ZMod& Z, const Mat& A, const Mat& B) {
    if (A.r != B.r || A.c != B.c) return false;
    for (size_t i = 0; i < A.a.size(); ++i)
        if (Z.red(A.a[i]) != Z.red(B.a[i])) return false;
    return true;
}
DenseBuf<uint32_t> ToDense(const Mat& A) {
    DenseBuf<uint32_t> D(A.r, A.c);
    for (uint32_t i = 0; i < A.r; ++i)
        for (uint32_t j = 0; j < A.c; ++j) D.at(i, j) = A.at(i, j);
    return D;
}
uint32_t VEll(uint64_t x, uint32_t ell) {
    uint32_t v = 0;
    while (x && x % ell == 0) {
        x /= ell;
        ++v;
    }
    return v;
}
AbelianGroupType PrimaryWithFree(const AbelianGroupType& G, uint32_t ell) {
    AbelianGroupType P = G.primary_part(ell);
    P.free_rank = G.free_rank;
    return P;
}

const LocalModel& FreeModel() { return FreeIsoModel(); }

}  // namespace

const LocalModel& FreeIsoModel() {
    static const LocalModel m(Descriptor5(), 0);
    return m;
}

Mat FromDenseBuf(const DenseBuf<uint32_t>& D) {
    Mat A(D.rows, D.cols);
    for (uint32_t i = 0; i < D.rows; ++i)
        for (uint32_t j = 0; j < D.cols; ++j) A.at(i, j) = D.at(i, j);
    return A;
}

namespace {
bool AllReduced(const ZMod& Z, const Mat& A) {
    if (Z.two) return true;   // exact mod 2^32, then masked: any representative
    for (uint32_t x : A.a)
        if (x >= Z.m) return false;
    return true;
}
}  // namespace

// A (r x n) * B (n x c), entries reduced mod Z.m (D2's maps::Mul, repeated here: that one is file-local to DescentMaps.cpp).
// (PAR-ASM, lever HECKE_DESCENT_PAR_ASM) reduced operands and a large product: maps::MulMod, the threaded SIMD product of PAR-D2,
// entrywise the same exact values (Z/ell^k is a ring; both return the reduced representative).  Unreduced operands at odd ell keep
// the serial code below (its accumulator bound assumes reduced inputs; MulMod's Montgomery path requires them).
Mat MatMulZ(const ZMod& Z, const Mat& A, const Mat& B) {
    ENGINE_CHECK_MSG(A.c == B.r, "pair::MatMulZ: shape");
    if (par::Enabled(par::Lever::Asm) && !par::InRegion() && uint64_t(A.r) * A.c * B.c >= (uint64_t(1) << 22) && AllReduced(Z, A) && AllReduced(Z, B))
        return maps::MulMod(Z, A, B);
    Mat C(A.r, B.c);
    if (A.r == 0 || B.c == 0) return C;
    if (Z.two) {
        for (uint32_t i = 0; i < A.r; ++i) {
            uint32_t* c = C.row(i);
            const uint32_t* a = A.row(i);
            for (uint32_t k = 0; k < A.c; ++k) {
                const uint32_t x = a[k];
                if (!x) continue;
                const uint32_t* b = B.row(k);
                for (uint32_t j = 0; j < B.c; ++j) c[j] += x * b[j];
            }
            for (uint32_t j = 0; j < B.c; ++j) c[j] &= Z.mask;
        }
        return C;
    }
    std::vector<uint64_t> acc(B.c);
    for (uint32_t i = 0; i < A.r; ++i) {
        std::fill(acc.begin(), acc.end(), 0);
        const uint32_t* a = A.row(i);
        uint32_t cnt = 0;
        for (uint32_t k = 0; k < A.c; ++k) {
            const uint64_t x = Z.red(a[k]);
            if (!x) continue;
            const uint32_t* b = B.row(k);
            for (uint32_t j = 0; j < B.c; ++j) acc[j] += x * Z.red(b[j]);
            if (++cnt == 3) {
                for (auto& y : acc) y %= Z.m;
                cnt = 0;
            }
        }
        uint32_t* c = C.row(i);
        for (uint32_t j = 0; j < B.c; ++j) c[j] = static_cast<uint32_t>(acc[j] % Z.m);
    }
    return C;
}

Mat InducedMatrix(const Pres& X, const Pres& Y, const Mat& f, ZMod& Zout) {
    ENGINE_CHECK_MSG(f.r == Y.c && f.c == X.c, "pair::InducedMatrix: the cochain map has the wrong shape");
    ENGINE_CHECK_MSG(X.Z.ell == Y.Z.ell, "pair::InducedMatrix: two primes");
    const ZMod Z = X.Z.k <= Y.Z.k ? X.Z : Y.Z;
    Zout = Z;
    const Mat T1 = MatMulZ(Z, Reduce(f, Z), Transpose(Reduce(X.gensT, Z)));
    Mat F = MatMulZ(Z, Reduce(Y.coord, Z), T1);
    for (uint32_t i = 0; i < Y.tors.size(); ++i) {
        const ZMod Zi(Z.ell, Y.tors[i]);
        for (uint32_t j = 0; j < F.c; ++j) F.at(i, j) = Zi.red(F.at(i, j));
    }
    return F;
}

uint64_t PairComplex::bytes() const {
    uint64_t b = 0;
    for (const auto* v : {&d, &db, &r})
        for (const Mat& m : *v) b += m.bytes();
    return b;
}

uint32_t ConeRank(const PairComplex& X, int j) {
    if (j < -1 || j + 1 > static_cast<int>(X.T)) return 0;
    return X.c[static_cast<uint32_t>(j + 1)] + (j >= 0 ? X.cb[static_cast<uint32_t>(j)] : 0u);
}

// D^j (c, c_d) = (-d c, d_d c_d + r c), c in C^{j+1}, c_d in C_d^j; rows C^{j+2} (+) C_d^{j+1}.  Needs j + 2 <= T.
Mat ConeDifferential(const PairComplex& X, int j) {
    ENGINE_CHECK_MSG(j >= -1 && j + 2 <= static_cast<int>(X.T), "pair::ConeDifferential: degree out of range");
    const uint32_t a = static_cast<uint32_t>(j + 1);   // bulk degree of the source
    const uint32_t tA = X.c[a + 1];
    Mat Dm(ConeRank(X, j + 1), ConeRank(X, j));
    for (uint32_t i = 0; i < X.d[a].r; ++i) {   // -d, placed directly (the values of Place(Neg(d)))
        const uint32_t* src = X.d[a].row(i);
        uint32_t* dst = Dm.row(i);
        for (uint32_t k = 0; k < X.d[a].c; ++k) dst[k] = X.Z.sub(0, X.Z.red(src[k]));
    }
    Place(Dm, X.r[a], tA, 0);
    if (j >= 0) Place(Dm, X.db[static_cast<uint32_t>(j)], tA, X.c[a]);
    return Dm;
}

// ---------------------------------------------------------------------------
// cusps and the closed form (prop:descentpair(3), (4))
// ---------------------------------------------------------------------------
std::vector<uint32_t> CuspWidths(uint32_t N) {
    const hecke::level::Gamma0LevelN A(N);
    const uint32_t m = A.m();
    std::vector<char> seen(m + 1, 0);
    std::vector<uint32_t> w;
    uint32_t total = 0;
    for (uint32_t j = 1; j <= m; ++j) {
        if (seen[j]) continue;
        uint32_t len = 0, k = j;
        do {
            seen[k] = 1;
            ++len;
            k = A.class_of(A.rep(k) * Mat22::T());
            ENGINE_CHECK_MSG(len <= m, "CuspWidths: a T-orbit longer than the index");
        } while (k != j);
        w.push_back(len);
        total += len;
    }
    ENGINE_CHECK_MSG(total == m, "CuspWidths: the widths do not add up to m_N (prop:descentpair(3))");
    std::sort(w.begin(), w.end());
    return w;
}

uint32_t Mod2InvariantDim(uint32_t n, uint32_t w) { return w % 2 == 0 ? n + 1 : n / 2 + 1; }

uint32_t ValuationFactorialPower(uint32_t n, uint32_t w, uint32_t ell) {
    uint32_t v = 0;
    for (uint64_t pk = ell; pk <= n; pk *= ell) v += static_cast<uint32_t>(n / pk);
    return v + n * VEll(w, ell);
}

std::vector<uint32_t> CoinvariantTorsionExponents(uint32_t n, uint32_t w, uint32_t ell, uint32_t k, bool& certified) {
    const ZMod Z(ell, k);
    const U32Ctx ctx = k >= 2 ? U32Ctx::prime_power(ell, k) : U32Ctx(ell);
    Mat Tw = FromDenseBuf(integral_model::rho_n(ctx, Mat22{1, static_cast<int64_t>(w), 0, 1}, n));
    for (uint32_t i = 0; i <= n; ++i) Tw.at(i, i) = Z.sub(Tw.at(i, i), 1);
    const maps::Smith S = maps::SmithForm(Z, Tw, false, false);
    certified = S.rank() == n;
    std::vector<uint32_t> ex;
    for (uint32_t s : S.sigma)
        if (s >= 1) ex.push_back(s);
    std::sort(ex.rbegin(), ex.rend());
    return ex;
}

AbelianGroupType BoundaryClosedForm(const std::vector<uint32_t>& widths, uint32_t n, uint32_t ell, uint32_t q, uint32_t k, bool& ok) {
    ok = true;
    AbelianGroupType G;
    const uint32_t c = static_cast<uint32_t>(widths.size());
    uint32_t r2 = 0;
    for (uint32_t w : widths) r2 += Mod2InvariantDim(n, w);
    if (n % 2 == 0) {
        if (q == 0) {
            G.free_rank = c;
        } else if (q == 1) {
            G.free_rank = c;
            for (uint32_t w : widths) {
                bool cert = false;
                for (uint32_t a : CoinvariantTorsionExponents(n, w, ell, k, cert)) G.add_torsion(ell, a);
                ok = ok && cert;
            }
        } else if (ell == 2) {
            for (uint32_t i = 0; i < r2; ++i) G.add_torsion(2, 1);
        }
    } else if (q >= 1 && ell == 2) {
        for (uint32_t i = 0; i < r2; ++i) G.add_torsion(2, 1);
    }
    return G;
}

PairBetti PairBettiOf(uint32_t N, uint32_t n) {
    PairBetti b;
    const LevelNCounts cf = LevelNClosedForm(N, n);
    b.b0 = cf.b0;
    b.b1 = cf.b1;
    b.c = static_cast<uint32_t>(CuspWidths(N).size());
    b.bd0 = b.bd1 = (n % 2 == 0) ? b.c : 0u;
    ENGINE_CHECK_MSG(b.bd1 >= b.b0 && b.b1 + b.b0 >= b.bd0, "PairBettiOf: the rational ranks of prop:descentpair(5)(b) are negative");
    return b;
}

// ---------------------------------------------------------------------------
// the pair complexes
// ---------------------------------------------------------------------------
namespace {

// The boundary of a pair given on a cell layout: db = the boundary block of d (after checking that d's boundary rows vanish
// off the boundary columns -- the subcomplex property, [C, IM2], in cochain form), r = the selection.
void SplitBoundary(PairComplex& X, const std::vector<std::vector<uint32_t>>& brows /* per degree: cochain rows of boundary cells */) {
    X.cb.clear();
    X.db.clear();
    X.r.clear();
    for (uint32_t q = 0; q <= X.T; ++q) X.cb.push_back(static_cast<uint32_t>(brows[q].size()));
    for (uint32_t q = 0; q <= X.T; ++q) {
        Mat R(X.cb[q], X.c[q]);
        for (uint32_t i = 0; i < brows[q].size(); ++i) R.at(i, brows[q][i]) = 1;
        X.r.push_back(std::move(R));
    }
    for (uint32_t q = 0; q < X.T; ++q) {
        const Mat& d = X.d[q];
        std::vector<char> isb(X.c[q], 0);
        for (uint32_t x : brows[q]) isb[x] = 1;
        for (uint32_t i : brows[q + 1]) {
            const uint32_t* row = d.row(i);
            for (uint32_t j = 0; j < d.c; ++j)
                ENGINE_CHECK_MSG(isb[j] || X.Z.red(row[j]) == 0,
                                 "pair: the boundary rows of d leave the boundary columns (Isob not a subcomplex, [C, IM2])");
        }
        X.db.push_back(Sub(d, brows[q + 1], brows[q]));
    }
}

}  // namespace

PairComplex BuildModelPair(const LocalModel& L, const U32Ctx& ctx, const CoefficientModule<uint32_t>& M, uint32_t T,
                           bool with_embedding) {
    ENGINE_CHECK_MSG(!ctx.is_field() && ctx.ell == L.ell(), "BuildModelPair: the context must be Z/ell^K at the model's ell");
    ENGINE_CHECK_MSG(T >= 1 && T <= L.q_top(), "BuildModelPair: T out of the model's range");
    PairComplex X;
    X.ell = L.ell();
    X.Z = ZMod(ctx.ell, ctx.k);
    X.T = T;
    X.route = L.ell() >= 5 ? "Bredon" : (L.ell() == 2 ? "Iso2" : "Iso3");
    const ZMod& Z = X.Z;
    std::map<integral_model::GKey, DenseBuf<uint32_t>> memo;
    std::map<std::pair<uint32_t, uint32_t>, std::pair<std::vector<uint32_t>, std::vector<uint32_t>>> sel;
    std::map<std::pair<uint32_t, uint32_t>, Mat> ginv;
    std::vector<std::vector<std::vector<uint32_t>>> J(T + 1), R(T + 1);
    std::vector<std::vector<const integral_model::ColumnSpec*>> col(T + 1);
    std::vector<std::vector<uint32_t>> off(T + 1), brows(T + 1);
    std::vector<std::vector<std::pair<uint32_t, uint32_t>>> key(T + 1);
    for (uint32_t q = 0; q <= T; ++q) {
        uint32_t c = 0;
        for (const auto& cell : L.cells(q)) {
            const auto k = std::make_pair(cell.s, cell.rep);
            const auto& cs = L.column(cell.s, cell.rep);
            auto it = sel.find(k);
            if (it == sel.end()) {
                std::vector<uint32_t> Jk, Rk;
                SelectBlockJR(ctx, cs, M, memo, Jk, Rk);
                if (cs.q_order > 1) {
                    const DenseBuf<uint32_t> P = integral_model::RhoQ(ctx, cs.e, M, memo);
                    Mat G(static_cast<uint32_t>(Rk.size()), static_cast<uint32_t>(Jk.size()));
                    for (uint32_t a = 0; a < Rk.size(); ++a)
                        for (uint32_t b = 0; b < Jk.size(); ++b) G.at(a, b) = P.at(Rk[a], Jk[b]);
                    // G = P[R, J] is invertible mod ell (the rank profiles); its inverse by Gauss-Jordan
                    const uint32_t w = G.r;
                    ENGINE_CHECK_MSG(G.r == G.c, "BuildModelPair: the block P[R, J] is not square");
                    Mat I = Identity(w);
                    for (uint32_t t = 0; t < w; ++t) {
                        uint32_t p = t;
                        while (p < w && Z.val(G.at(p, t)) != 0) ++p;
                        ENGINE_CHECK_MSG(p < w, "BuildModelPair: the block P[R, J] is not invertible mod ell");
                        if (p != t)
                            for (uint32_t jj = 0; jj < w; ++jj) {
                                std::swap(G.at(t, jj), G.at(p, jj));
                                std::swap(I.at(t, jj), I.at(p, jj));
                            }
                        const uint32_t u = Z.inv(G.at(t, t));
                        for (uint32_t jj = 0; jj < w; ++jj) {
                            G.at(t, jj) = Z.mul(G.at(t, jj), u);
                            I.at(t, jj) = Z.mul(I.at(t, jj), u);
                        }
                        for (uint32_t i = 0; i < w; ++i) {
                            if (i == t || !G.at(i, t)) continue;
                            const uint32_t f = G.at(i, t);
                            for (uint32_t jj = 0; jj < w; ++jj) {
                                G.at(i, jj) = Z.sub(G.at(i, jj), Z.mul(f, G.at(t, jj)));
                                I.at(i, jj) = Z.sub(I.at(i, jj), Z.mul(f, I.at(t, jj)));
                            }
                        }
                    }
                    ginv.emplace(k, std::move(I));
                }
                it = sel.emplace(k, std::make_pair(Jk, Rk)).first;
            }
            J[q].push_back(it->second.first);
            R[q].push_back(it->second.second);
            col[q].push_back(&cs);
            off[q].push_back(c);
            key[q].push_back(k);
            if (cell.boundary)
                for (uint32_t a = 0; a < it->second.first.size(); ++a) brows[q].push_back(c + a);
            c += static_cast<uint32_t>(it->second.first.size());
        }
        X.c.push_back(c);
    }
    for (uint32_t q = 0; q < T; ++q) {
        Mat A = FromDenseBuf(ClauseMatrixReduced(L, ctx, M, q, J[q], R[q + 1], col[q], col[q + 1]));
        ENGINE_CHECK(A.r == X.c[q + 1] && A.c == X.c[q]);
        // pi_R coordinates -> J coordinates on the target (D2's BuildRed): G_y^{-1} per target cell with Q != 1
        for (uint32_t y = 0; y < col[q + 1].size(); ++y) {
            if (col[q + 1][y]->q_order == 1) continue;
            const Mat& Gi = ginv.at(key[q + 1][y]);
            const uint32_t o = off[q + 1][y], w = Gi.r;
            Mat blk(w, A.c);
            for (uint32_t a = 0; a < w; ++a) std::copy(A.row(o + a), A.row(o + a) + A.c, blk.row(a));
            const Mat nb = MatMulZ(Z, Gi, blk);
            for (uint32_t a = 0; a < w; ++a) std::copy(nb.row(a), nb.row(a) + A.c, A.row(o + a));
        }
        X.d.push_back(std::move(A));
    }
    SplitBoundary(X, brows);
    if (with_embedding) {
        const uint32_t D = M.dim;
        for (uint32_t q = 0; q <= T; ++q) {
            const uint32_t W = static_cast<uint32_t>(col[q].size());
            Mat E(W * D, X.c[q]), Cq(X.c[q], W * D);
            for (uint32_t x = 0; x < W; ++x) {
                const auto& Jx = J[q][x];
                const auto& Rx = R[q][x];
                const uint32_t o = off[q][x];
                if (col[q][x]->q_order == 1) {
                    for (uint32_t b = 0; b < Jx.size(); ++b) {
                        E.at(x * D + Jx[b], o + b) = 1;
                        Cq.at(o + b, x * D + Jx[b]) = 1;
                    }
                    continue;
                }
                const DenseBuf<uint32_t> P = integral_model::RhoQ(ctx, col[q][x]->e, M, memo);
                for (uint32_t i = 0; i < D; ++i)
                    for (uint32_t b = 0; b < Jx.size(); ++b) E.at(x * D + i, o + b) = Z.red(P.at(i, Jx[b]));
                const Mat& Gi = ginv.at(key[q][x]);
                for (uint32_t a = 0; a < Gi.r; ++a)
                    for (uint32_t b = 0; b < Rx.size(); ++b) Cq.at(o + a, x * D + Rx[b]) = Gi.at(a, b);
            }
            ENGINE_CHECK_MSG(EqualMat(Z, MatMulZ(Z, Cq, E), Identity(X.c[q])), "BuildModelPair: coords o emb != 1");
            X.emb.push_back(std::move(E));
            X.coords.push_back(std::move(Cq));
        }
    }
    return X;
}

PairComplex BuildFreePair(const U32Ctx& ctx, const CoefficientModule<uint32_t>& M, uint32_t T) {
    const LocalModel& L = FreeModel();
    ENGINE_CHECK_MSG(T >= 1 && T <= L.q_top(), "BuildFreePair: T out of the model's range");
    PairComplex X;
    X.ell = static_cast<uint32_t>(ctx.ell);
    X.Z = ctx.is_field() ? ZMod(static_cast<uint32_t>(ctx.p), 1) : ZMod(static_cast<uint32_t>(ctx.ell), ctx.k);
    X.T = T;
    X.route = "IsoFree";
    const uint32_t D = M.dim;
    std::vector<std::vector<uint32_t>> brows(T + 1);
    for (uint32_t q = 0; q <= T; ++q) {
        const auto& cells = L.cells(q);
        X.c.push_back(static_cast<uint32_t>(cells.size()) * D);
        for (uint32_t i = 0; i < cells.size(); ++i)
            if (cells[i].boundary)
                for (uint32_t a = 0; a < D; ++a) brows[q].push_back(i * D + a);
    }
    for (uint32_t q = 0; q < T; ++q) X.d.push_back(FromDenseBuf(L.coboundary(ctx, M, q)));
    SplitBoundary(X, brows);
    // the boundary block is the model's own boundary coboundary (LocalModel::boundary_coboundary), entry for entry
    for (uint32_t q = 0; q < T; ++q)
        ENGINE_CHECK_MSG(EqualMat(X.Z, X.db[q], FromDenseBuf(L.boundary_coboundary(ctx, M, q))),
                         "BuildFreePair: the boundary block differs from LocalModel::boundary_coboundary");
    return X;
}

PairComplex BuildFreePairN(uint32_t N, uint32_t n, const U32Ctx& ctx, uint32_t T) {
    return BuildFreePair(ctx, InducedSymModule(N, n, ctx), T);
}

// ---------------------------------------------------------------------------
// the shipped E8 item-4b pair (level 1, ell >= 5)
// ---------------------------------------------------------------------------
PairComplex BuildE8Pair(uint32_t n, uint32_t ell, uint32_t K, uint32_t T, E8Identity* id) {
    ENGINE_CHECK_MSG(ell >= 5 && IsPrime(ell) && K >= 2, "BuildE8Pair: ell >= 5 prime, K >= 2 (the shipped module route)");
    hecke::ScopedCoefficientRing ring(K);
    const U32Ctx ctx = U32Ctx::prime_power(ell, K);
    SimplicialComplex K7 = CreateSL2ZFundamentalDomainTriangulation();
    LocalSystemSimplicialComplex lssc(K7, hecke::GenerateSL2ZRelationsRing<uint32_t>(K7, n, ctx), n + 1, ctx.p);
    EquivariantCohomologyOnK coh(&lssc);
    coh.compute_cohomology();
    hecke::SL2ZBoundaryPair pe(&coh, K7);
    const EquivariantCohomologyOnK& cb = pe.boundary_cohomology();
    PairComplex X;
    X.ell = ell;
    X.Z = ZMod(ell, K);
    X.T = T;
    X.route = "E8pair";
    const uint32_t D = n + 1;
    std::vector<Mat> Cb(T + 1), Ck(T + 1), Pfl(T + 1);
    for (uint32_t q = 0; q <= T; ++q) {
        if (q <= 2) {
            Ck[q] = FromDenseBuf(coh.module_degree(q).C);
            X.c.push_back(Ck[q].c);
        } else {
            X.c.push_back(0);
        }
        if (q <= 1) {
            Cb[q] = FromDenseBuf(cb.module_degree(q).C);
            X.cb.push_back(Cb[q].c);
        } else {
            X.cb.push_back(0);
        }
    }
    for (uint32_t q = 0; q < T; ++q) {
        if (q <= 1) X.d.push_back(FromDenseBuf(coh.module_X(q)));
        else X.d.push_back(Mat(X.c[q + 1], X.c[q]));
        ENGINE_CHECK(X.d.back().r == X.c[q + 1] && X.d.back().c == X.c[q]);
        if (q == 0) X.db.push_back(FromDenseBuf(cb.module_X(0)));
        else X.db.push_back(Mat(X.cb[q + 1], X.cb[q]));
        ENGINE_CHECK(X.db.back().r == X.cb[q + 1] && X.db.back().c == X.cb[q]);
    }
    // the restriction in the module bases: P_flat C_K, read in the boundary basis through its unit Howell pivots (C[piv, :] = I)
    for (uint32_t q = 0; q <= T; ++q) {
        Mat R(X.cb[q], X.c[q]);
        if (q <= 1) {
            Pfl[q] = FromDenseBuf(pe.restriction().pullback_matrix_module(q));
            const Mat Y = MatMulZ(X.Z, Pfl[q], Ck[q]);   // boundary flat x c_q
            const auto& piv = cb.module_degree(q).piv;
            ENGINE_CHECK_MSG(piv.size() == X.cb[q], "BuildE8Pair: the circle model's basis has no unit pivots (non-unit branch)");
            for (uint32_t i = 0; i < piv.size(); ++i) std::copy(Y.row(piv[i]), Y.row(piv[i]) + Y.c, R.row(i));
            ENGINE_CHECK_MSG(EqualMat(X.Z, MatMulZ(X.Z, Cb[q], R), Y), "BuildE8Pair: P_flat C_K does not lie in the circle model's span");
        }
        X.r.push_back(std::move(R));
    }
    if (id) {
        // the shipped module cone
        const std::unique_ptr<EquivariantMappingCone> cone = pe.make_cone();
        const auto& mc = cone->module_cone();
        id->shipped_Hc_invariant_factors.clear();
        for (uint32_t q = 0; q <= 2; ++q) id->shipped_Hc_invariant_factors.push_back(mc.H_invariant_factors(q));
        // the identity of complexes through cmpI^* = eps^* (prop:descentpair(1)(d))
        const LocalModel& L = BredonModel(ell);
        const integral_model::IsotropyModel& desc = L.descriptor();
        const CoefficientModule<uint32_t> Ms = integral_model::SymModule(ctx, n);
        const auto& Ksim = integral_model::IsotropyModel::K_simplices();
        const auto& Kbd = pe.K_bd().get_simplices();
        const ZMod& Z = X.Z;
        std::vector<Mat> E(3), Eb(2), sel(3);
        for (uint32_t q = 0; q <= 2; ++q) {
            const auto& cells = L.cells(q);
            E[q] = Mat(static_cast<uint32_t>(cells.size()) * D, static_cast<uint32_t>(Ksim[q].size()) * D);
            uint32_t nb = 0;
            for (const auto& cl : cells) nb += cl.boundary ? 1 : 0;
            sel[q] = Mat(nb * D, static_cast<uint32_t>(cells.size()) * D);
            if (q <= 1) Eb[q] = Mat(nb * D, static_cast<uint32_t>(Kbd[q].size()) * D);
            uint32_t bi = 0;
            for (uint32_t i = 0; i < cells.size(); ++i) {
                ENGINE_CHECK_MSG(cells[i].t == 0 && cells[i].s == q, "BuildE8Pair: a periodic column in the Bredon model");
                const auto& rep = desc.reps(q)[cells[i].rep];
                const uint32_t k = static_cast<uint32_t>(std::find(Ksim[q].begin(), Ksim[q].end(), rep) - Ksim[q].begin());
                ENGINE_CHECK(k < Ksim[q].size());
                for (uint32_t a = 0; a < D; ++a) E[q].at(i * D + a, k * D + a) = 1;
                if (cells[i].boundary) {
                    for (uint32_t a = 0; a < D; ++a) sel[q].at(bi * D + a, i * D + a) = 1;
                    if (q <= 1) {
                        uint32_t kb = UINT32_MAX;
                        for (uint32_t s = 0; s < Kbd[q].size(); ++s)
                            if (Kbd[q][s] == rep) kb = s;
                        ENGINE_CHECK_MSG(kb != UINT32_MAX, "BuildE8Pair: a boundary representative is not a simplex of K_bd");
                        for (uint32_t a = 0; a < D; ++a) Eb[q].at(bi * D + a, kb * D + a) = 1;
                    }
                    ++bi;
                }
            }
        }
        bool bulk = true, bd = true, rs = true, onto = true;
        for (uint32_t q = 0; q <= 2; ++q) {
            const Mat EC = MatMulZ(Z, E[q], Ck[q]);
            const Mat P = FromDenseBuf(L.projector(ctx, Ms, q));
            onto = onto && EqualMat(Z, MatMulZ(Z, P, EC), EC);
            const uint32_t lEC = EC.c ? engine::elim::length<uint32_t>(ctx, engine::cview(ToDense(EC))) : 0u;
            const uint32_t lP = P.c ? engine::elim::length<uint32_t>(ctx, engine::cview(ToDense(P))) : 0u;
            onto = onto && lEC == lP && lEC == X.c[q] * K;
            if (q < 2) {
                const Mat lhs = MatMulZ(Z, FromDenseBuf(L.coboundary(ctx, Ms, q)), EC);
                const Mat rhs = MatMulZ(Z, MatMulZ(Z, E[q + 1], Ck[q + 1]), X.d[q]);
                bulk = bulk && EqualMat(Z, lhs, rhs);
                id->entries_checked += lhs.a.size();
            }
            if (q <= 1) {
                const Mat EbC = MatMulZ(Z, Eb[q], Cb[q]);
                const Mat Pb = FromDenseBuf(L.boundary_projector(ctx, Ms, q));
                onto = onto && EqualMat(Z, MatMulZ(Z, Pb, EbC), EbC);
                const uint32_t lb = EbC.c ? engine::elim::length<uint32_t>(ctx, engine::cview(ToDense(EbC))) : 0u;
                const uint32_t lPb = Pb.c ? engine::elim::length<uint32_t>(ctx, engine::cview(ToDense(Pb))) : 0u;
                onto = onto && lb == lPb && lb == X.cb[q] * K;
                if (q == 0) {
                    const Mat lhs = MatMulZ(Z, FromDenseBuf(L.boundary_coboundary(ctx, Ms, 0)), EbC);
                    const Mat rhs = MatMulZ(Z, MatMulZ(Z, Eb[1], Cb[1]), X.db[0]);
                    bd = bd && EqualMat(Z, lhs, rhs);
                    id->entries_checked += lhs.a.size();
                }
                // restriction: sel eps^* C_K = eps_d^* (P_flat C_K)
                const Mat lhs = MatMulZ(Z, sel[q], MatMulZ(Z, E[q], Ck[q]));
                const Mat rhs = MatMulZ(Z, Eb[q], MatMulZ(Z, Pfl[q], Ck[q]));
                rs = rs && EqualMat(Z, lhs, rhs);
                id->entries_checked += lhs.a.size();
            }
        }
        id->bulk_chain_map = bulk;
        id->bdry_chain_map = bd;
        id->res_compatible = rs;
        id->onto = onto;
        if (!id->all())
            id->why = std::string(bulk ? "" : " bulk eps^* not a chain map;") + (bd ? "" : " boundary eps^* not a chain map;") +
                      (rs ? "" : " restriction not compatible;") + (onto ? "" : " eps^* not onto the Bredon cochains;");
    }
    return X;
}

// ---------------------------------------------------------------------------
// the sequence on one piece
// ---------------------------------------------------------------------------
bool RunPairLes(const PairComplex& X, const PairBetti& bt, uint32_t qm, PairLocal& L, PieceReport& rep, int32_t test_rank_offset) {
    const ZMod& Z = X.Z;
    const uint32_t T = X.T;
    ENGINE_CHECK_MSG(qm + 1 <= T, "RunPairLes: the reported degrees need cochains to qm + 1");
    auto fail = [&](const std::string& w) {
        rep.ok = false;
        rep.why = w;
        return false;
    };
    // rational ranks by the Betti recursions (prop:descentpair(6): bulk from b_q, boundary from the closed form, cone from (5)(b))
    std::vector<uint32_t> rd, rdb, rc;
    {
        int64_t p = 0, pb = 0;
        for (uint32_t q = 0; q < T; ++q) {
            const int64_t x = int64_t(X.c[q]) - int64_t(bt.b(q)) - p;
            const int64_t y = int64_t(X.cb[q]) - int64_t(bt.bd(q)) - pb;
            if (x < 0 || y < 0) return fail("negative Betti-recursion rank at q = " + std::to_string(q));
            rd.push_back(static_cast<uint32_t>(x));
            rdb.push_back(static_cast<uint32_t>(y));
            p = x;
            pb = y;
        }
        int64_t pc = 0;
        for (int j = -1; j + 2 <= static_cast<int>(T); ++j) {
            const int64_t x = int64_t(ConeRank(X, j)) - int64_t(bt.bc(j + 1)) - pc;
            if (x < 0) return fail("negative cone Betti-recursion rank at j = " + std::to_string(j));
            rc.push_back(static_cast<uint32_t>(x));
            pc = x;
        }
        // (6)(b): rank_Q D^{-1} = rank P(M)^0 (no source needed): the recursion must agree
        if (rc[0] != X.c[0]) return fail("rank_Q D^{-1} != rank C^0 (prop:descentpair(5)(a), (6)(b))");
    }
    // rres is a cochain map (d_d r = r d), so D D = 0 on the cone (eq:conediff): checked entrywise, transposed so that the sparse
    // selection r^T is the left factor
    // (PAR-D2: on the lever HECKE_DESCENT_SMITH_BLOCKED the same entrywise identity r d = d_d r untransposed, streamed in row blocks
    // and threaded, no product materialised: maps::ProductsEqual)
    const bool blk = maps::DescentSmithBlockedEnabled();
    for (uint32_t q = 0; q < T; ++q) {
        bool same;
        if (blk) {
            same = maps::ProductsEqual(Z, X.r[q + 1], X.d[q], X.db[q], X.r[q]);
        } else {
            const Mat lhs = Transpose(MatMulZ(Z, X.r[q + 1], X.d[q]));
            const Mat rhs = MatMulZ(Z, Transpose(X.r[q]), Transpose(X.db[q]));
            same = EqualMat(Z, lhs, rhs);
        }
        if (!same) return fail("rres is not a cochain map at q = " + std::to_string(q) + " (d_d r != r d; the cone is no complex)");
        ++rep.checks;
    }
    // D5c: the cone differentials are built on demand and only two are alive at a time (D^{q-2} and D^{q-1} for H^q_c): the memory of
    // the tree pair at large D (DESCENT5c_report.md); the values are those of ConeDifferential, unchanged
    Mat Dprev, Dnext;   // D^{q-2}, D^{q-1}
    L.ell = X.ell;
    L.qm = qm;
    const auto tl = clk::now();
    for (uint32_t q = 0; q <= qm; ++q) {
        L.H.push_back(BuildPres(Z, q ? X.d[q - 1] : Mat(X.c[q], 0), q ? rd[q - 1] : 0u, X.d[q], X.c[q], bt.b(q)));
        L.Hd.push_back(BuildPres(Z, q ? X.db[q - 1] : Mat(X.cb[q], 0), q ? rdb[q - 1] : 0u, X.db[q], X.cb[q], bt.bd(q)));
        // H^q_c = H^{q-1}(Cone): previous D^{q-2} (index q - 1), next D^{q-1} (index q)
        if (q >= 1) Dprev = std::move(Dnext);
        Dnext = ConeDifferential(X, static_cast<int>(q) - 1);
        Mat prev = q >= 1 ? std::move(Dprev) : Mat(ConeRank(X, -1), 0);
        Dprev = Mat();
        const uint32_t rprev = q >= 1 ? rc[q - 1] : 0u;
        // (PAR-D2) consumed: D^{q-2} is not used again
        L.Hc.push_back(BuildPres(Z, std::move(prev), rprev, Dnext, ConeRank(X, static_cast<int>(q) - 1), bt.bc(static_cast<int>(q))));
        for (const Pres* P : {&L.H.back(), &L.Hd.back(), &L.Hc.back()}) {
            if (!P->ok) {
                const char* which = P == &L.H.back() ? "H" : (P == &L.Hd.back() ? "H(d)" : "H_c");
                return fail(std::string("presentation of ") + which + "^" + std::to_string(q) + ": " + P->why);
            }
            rep.min_prec = std::min(rep.min_prec, P->Z.k);
        }
    }
    for (uint32_t q = 0; q <= qm; ++q) {
        const uint32_t rr = q == 1 ? uint32_t(int64_t(bt.rank_res(q)) + test_rank_offset) : bt.rank_res(q);
        L.res.push_back(InducedMap(L.H[q], L.Hd[q], X.r[q], rr));
        // (PAR-ASM) j and proj are identity-like: maps::InducedMapEmbed (entrywise InducedMap with the dense f; lever
        // HECKE_DESCENT_EMBED_MAP=0 forms f as before)
        if (q < qm)   // iota_* = j : C_d^q -> Cone^q = C^{q+1} (+) C_d^q, c_d |-> (0, c_d)
            L.j.push_back(maps::InducedMapEmbed(L.Hd[q], L.Hc[q + 1], X.c[q + 1], 0, X.cb[q], bt.rank_j(q), ConeRank(X, static_cast<int>(q)),
                                                X.cb[q]));
        // pi_* = proj : Cone^{q-1} = C^q (+) C_d^{q-1} -> C^q, (c, c_d) |-> c.  SIGN: = -delta (prop:descentpair(1)(c), referee m4).
        L.proj.push_back(maps::InducedMapEmbed(L.Hc[q], L.H[q], 0, 0, X.c[q], bt.rank_proj(q), X.c[q], ConeRank(X, static_cast<int>(q) - 1)));
    }
    for (uint32_t q = 0; q <= qm; ++q) {
        if (!L.res[q].ok) return fail("res^" + std::to_string(q) + ": " + L.res[q].why);
        if (!L.proj[q].ok) return fail("proj^" + std::to_string(q) + ": " + L.proj[q].why);
        if (q < qm && !L.j[q].ok) return fail("iota^" + std::to_string(q) + ": " + L.j[q].why);
    }
    for (const auto* v : {&L.res, &L.j, &L.proj})
        for (const auto& m : *v) rep.min_prec = std::min(rep.min_prec, m.ZL.k ? m.ZL.k : m.Z.k);
    // exactness at every reported spot (code map (e)): H^0, then H^q(d), H^{q+1}_c, H^{q+1}
    std::string why;
    if (!L.Hc[0].type().is_zero()) return fail("H^0_c != 0 (prop:descentpair(1)(c))");
    if (!maps::ExactAt(L.H[0], L.proj[0], L.res[0], why)) return fail("not exact at H^0: " + why);
    ++rep.exact_spots;
    for (uint32_t q = 0; q < qm; ++q) {
        if (!maps::ExactAt(L.Hd[q], L.res[q], L.j[q], why)) return fail("not exact at H^" + std::to_string(q) + "(d): " + why);
        if (!maps::ExactAt(L.Hc[q + 1], L.j[q], L.proj[q + 1], why))
            return fail("not exact at H^" + std::to_string(q + 1) + "_c: " + why);
        if (!maps::ExactAt(L.H[q + 1], L.proj[q + 1], L.res[q + 1], why))
            return fail("not exact at H^" + std::to_string(q + 1) + ": " + why);
        rep.exact_spots += 3;
    }
    // the a-priori exponents (prop:descentpair(6)) and (5)(a), (c)
    const uint32_t ell = X.ell;
    const uint32_t cap48 = VEll(48, ell), cap2 = VEll(2, ell);
    for (uint32_t q = 0; q <= qm; ++q) {
        for (uint32_t a : L.Hd[q].tors)
            if (q >= 2 && a > cap2) return fail("an exponent of tors H^" + std::to_string(q) + "(d) exceeds v_ell(2)");
        if (q == 1 && !L.Hc[1].tors.empty()) return fail("H^1_c has torsion (prop:descentpair(1)(c), (5)(a))");
        for (uint32_t a : L.Hc[q].tors)
            if (q >= 3 && a > cap48) return fail("an exponent of tors H^" + std::to_string(q) + "_c exceeds v_ell(48) (prop:descentpair(5)(c))");
        if (L.H[q].free != bt.b(q) || L.Hd[q].free != bt.bd(q) || L.Hc[q].free != bt.bc(static_cast<int>(q)))
            return fail("a free rank at q = " + std::to_string(q) + " is off the closed forms");
        ++rep.checks;
    }
    // ell = 3: pi_* : H^q_c -> H^q is an isomorphism (x) Z_(3) for q >= 3 (prop:descentpair(5)(c)); ell >= 5: H^q_c has no
    // torsion for q != 2 ((5)(c))
    for (uint32_t q = 3; q <= qm; ++q) {
        if (ell == 3 && (!L.proj[q].ker.is_zero() || !L.proj[q].coker.is_zero()))
            return fail("ell = 3: pi_* on H^" + std::to_string(q) + "_c is not an isomorphism");
        if (ell >= 5 && !L.Hc[q].tors.empty()) return fail("ell >= 5: H^" + std::to_string(q) + "_c has torsion");
        ++rep.checks;
    }
    rep.les_seconds += since(tl);
    return true;
}

// ---------------------------------------------------------------------------
// the ell' rank source on the full layout (code map (c))
// ---------------------------------------------------------------------------
namespace {

// rank over F_ell' of the cone differentials D^j (j = -1 .. T - 2) of the O_ell-model L (nullptr: the free model) with fibre
// M_N, on the full layout restricted to the cochains: rank(D P), P = diag(P_{j+1}, P_d^j), applied block by block.
std::vector<uint32_t> ConeRanksModPrime(const LocalModel& L, uint32_t N, uint32_t n, uint32_t ellp, uint32_t T, bool free_model) {
    const U32Ctx F(ellp);
    const CoefficientModule<uint32_t> M = InducedSymModule(N, n, F);
    const uint32_t D = M.dim;
    // full layout cochains with the projector blocks per cell
    struct Deg {
        uint32_t w = 0;                    // cells
        std::vector<uint32_t> bcells;      // boundary cell indices
        std::vector<DenseBuf<uint32_t>> P; // per cell (empty: identity)
    };
    std::vector<Deg> deg(T + 1);
    std::map<integral_model::GKey, DenseBuf<uint32_t>> memo;
    for (uint32_t q = 0; q <= T; ++q) {
        const auto& cells = L.cells(q);
        deg[q].w = static_cast<uint32_t>(cells.size());
        for (uint32_t i = 0; i < cells.size(); ++i) {
            if (cells[i].boundary) deg[q].bcells.push_back(i);
            if (free_model) deg[q].P.emplace_back();
            else deg[q].P.push_back(integral_model::RhoQ(F, L.column(cells[i].s, cells[i].rep).e, M, memo));
        }
    }
    const ZMod Zp(ellp, 1);
    std::vector<uint32_t> ranks;
    for (int j = -1; j + 2 <= static_cast<int>(T); ++j) {
        const uint32_t a = static_cast<uint32_t>(j + 1);
        // columns: C^{a} cells, then boundary cells of degree j (as blocks of the full layout)
        std::vector<std::pair<uint32_t, uint32_t>> colblk;   // (degree, cell) and a tag for the boundary half
        const Mat d = FromDenseBuf(L.coboundary(F, M, a));        // C^{a+1} x C^{a}
        Mat db;
        if (j >= 0) db = FromDenseBuf(L.boundary_coboundary(F, M, static_cast<uint32_t>(j)));
        const uint32_t nb_src = j >= 0 ? static_cast<uint32_t>(deg[static_cast<uint32_t>(j)].bcells.size()) : 0u;
        const uint32_t nb_tgt = static_cast<uint32_t>(deg[a].bcells.size());
        const uint32_t rows = deg[a + 1].w * D + nb_tgt * D, cols = deg[a].w * D + nb_src * D;
        Mat Dm(rows, cols);
        // -d on the bulk
        for (uint32_t i = 0; i < d.r; ++i)
            for (uint32_t k = 0; k < d.c; ++k) Dm.at(i, k) = Zp.sub(0, d.at(i, k));
        // r: the boundary blocks of C^{a}
        for (uint32_t b = 0; b < nb_tgt; ++b)
            for (uint32_t x = 0; x < D; ++x) Dm.at(deg[a + 1].w * D + b * D + x, deg[a].bcells[b] * D + x) = 1;
        if (j >= 0) {
            ENGINE_CHECK(db.r == nb_tgt * D && db.c == nb_src * D);
            for (uint32_t i = 0; i < db.r; ++i)
                for (uint32_t k = 0; k < db.c; ++k) Dm.at(deg[a + 1].w * D + i, deg[a].w * D + k) = db.at(i, k);
        }
        // right-multiply by the block-diagonal projector of the source
        if (!free_model) {
            Mat Y(rows, cols);
            auto apply = [&](uint32_t col0, const DenseBuf<uint32_t>& P) {
                for (uint32_t i = 0; i < rows; ++i) {
                    const uint32_t* x = Dm.row(i) + col0;
                    uint32_t* y = Y.row(i) + col0;
                    for (uint32_t c = 0; c < D; ++c) {
                        uint64_t s = 0;
                        for (uint32_t k = 0; k < D; ++k) s += uint64_t(x[k]) * P.at(k, c) % ellp;
                        y[c] = static_cast<uint32_t>(s % ellp);
                    }
                }
            };
            for (uint32_t c = 0; c < deg[a].w; ++c) apply(c * D, deg[a].P[c]);
            for (uint32_t b = 0; b < nb_src; ++b) apply(deg[a].w * D + b * D, deg[static_cast<uint32_t>(j)].P[deg[static_cast<uint32_t>(j)].bcells[b]]);
            Dm = std::move(Y);
        }
        DenseBuf<uint32_t> Bd = ToDense(Dm);
        ranks.push_back(rows && cols ? engine::elim::rref(F, engine::view(Bd)).rank : 0u);
    }
    return ranks;
}

uint32_t QEllOf(uint32_t ell) { return ell == 2 ? 3u : (ell == 3 ? 4u : 12u); }

uint32_t DefaultRankPrime(uint32_t n) {
    uint32_t p = 10007;
    while (!(IsPrime(p) && p > n + 1)) ++p;
    return p;
}

}  // namespace

// ---------------------------------------------------------------------------
// RSS
// ---------------------------------------------------------------------------
uint64_t PairRssStructural(uint32_t ell, uint32_t D, uint32_t T) {
    ENGINE_CHECK(ell == 2 || ell == 3);
    // the reduced cochain ranks per unit of D (DescentAssembly's Sec. 3.1 figures; the boundary at 2 is free, 3, 6, 6, ... D;
    // at 3 the circle model, 3/2, 3/2 D in degrees 0, 1)
    static const double w2[] = {4.34, 13.34, 18.34, 18.34, 18.34, 18.34, 18.34};
    static const double w3[] = {4.5, 10.0, 6.0, 1.0, 1.0, 1.0, 1.0};
    static const double b2[] = {3.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0};
    static const double b3[] = {1.5, 1.5, 0, 0, 0, 0, 0};
    auto c = [&](uint32_t q) { return uint64_t((ell == 2 ? w2[q] : w3[q]) * D) + 1; };
    auto cb = [&](uint32_t q) { return uint64_t((ell == 2 ? b2[q] : b3[q]) * D) + 1; };
    uint64_t stored = 0, cmax = 0;
    for (uint32_t q = 0; q < T; ++q) stored += c(q + 1) * c(q) + cb(q + 1) * cb(q) + cb(q) * c(q);
    for (int j = -1; j + 2 <= static_cast<int>(T); ++j) {
        const uint64_t cj = c(static_cast<uint32_t>(j + 1)) + (j >= 0 ? cb(static_cast<uint32_t>(j)) : 0);
        const uint64_t cj1 = c(static_cast<uint32_t>(j + 2)) + cb(static_cast<uint32_t>(j + 1));
        stored += cj * cj1;
        cmax = std::max(cmax, std::max(cj, cj1));
    }
    // working: the largest Smith form with left transforms (A + L + L^{-1}) plus the right-transform Smith of the next
    // differential on the complement (b > 0 degrees), bounded by two more c x c; the rho memo (48 D^2).
    return 4ull * (stored + 5 * cmax * cmax) + 48ull * 4 * uint64_t(D) * D;
}
uint64_t PredictPairRss(uint32_t ell, uint32_t D, uint32_t T) {
    // calibration factor: DESCENT4_predictions.md (filed before the heavy runs)
    return PairRssStructural(ell, D, T) / 10 * 12 + (256ull << 20);
}

// ---------------------------------------------------------------------------
// the assembly
// ---------------------------------------------------------------------------
PairResult AssembleDescentPair(uint32_t N, uint32_t n, const PairOptions& opt) {
    const auto t0 = clk::now();
    PairResult R;
    R.N = N;
    R.n = n;
    R.q_max = opt.q_max;
    ENGINE_CHECK_MSG(N >= 1, "AssembleDescentPair: N >= 1");
    ENGINE_CHECK_MSG(opt.q_max >= 4, "AssembleDescentPair: q_max >= 4 (code map (g): q_max >= 4 for the pair)");
    if (opt.q_max > 5) {   // C-D7p referee m2: the pair is served for q_max <= 5 (the descriptor's and the pair lift's degree 6)
        R.refused = true;
        R.refusal = "REFUSED [DegreeRange] (N, n) = (" + std::to_string(N) + ", " + std::to_string(n) + "): q_max = " + std::to_string(opt.q_max) +
                    " > 5 (cochains and the pair lift stop at degree 6; higher degrees come from prop:descentpairhecke(7))";
        return R;
    }
    const uint32_t T = opt.q_max + 1;
    R.widths = CuspWidths(N);
    R.bt = PairBettiOf(N, n);
    R.D = PsiIndex(N) * (n + 1);
    const bool even = n % 2 == 0;
    std::set<uint32_t> S{2};
    if (even) {
        S.insert(3);
        for (uint32_t p = 5; p <= n + 1; ++p)
            if (IsPrime(p)) S.insert(p);
        if (n > 0)
            for (uint32_t p = 5; p <= N; ++p)
                if (IsPrime(p) && N % p == 0) S.insert(p);
        for (const auto& pr : BadPrimesN(N, n)) S.insert(pr.first);
    }
    for (uint32_t p : opt.extra_primes) {
        ENGINE_CHECK_MSG(IsPrime(p), "AssembleDescentPair: an extra prime is not prime");
        S.insert(p);
    }
    std::set<uint32_t> extra(opt.extra_primes.begin(), opt.extra_primes.end());
    R.primes.assign(S.begin(), S.end());
    std::ostringstream why;
    // rule 19: predict the dense pieces first, refuse above 0.9 x the ceiling before anything is built
    // D5c: the tree cusp pair is the route of the ell in {2, 3} pieces (code map (c) of sec:treepair) unless HECKE_DESCENT_TREE_PAIR=0
    // (or HECKE_DESCENT_TREE=0); read ONCE per call.  The free-model route (PairRoute::IsoFree) is unchanged.
    const bool tree_pair = opt.route == PairRoute::IsoEll && DescentTreePairEnabled();
    R.route_two = tree_pair ? "tree" : (opt.route == PairRoute::IsoFree ? "IsoFree" : "Iso2");
    if (tree_pair) {
        R.rss_pred = treepair::PredictTreePairRss(2, R.D, T);
        if (even) R.rss_pred = std::max(R.rss_pred, treepair::PredictTreePairRss(3, R.D, T));
    } else {
        R.rss_pred = PredictPairRss(2, R.D, T);
        if (even) R.rss_pred = std::max(R.rss_pred, PredictPairRss(3, R.D, T));
    }
    if (opt.route == PairRoute::IsoFree) R.rss_pred = R.rss_pred / 10 * 12;   // the full layout: 19 / 18.33 per side, and no projector
    if (R.rss_pred > opt.rss_cap_bytes / 10 * 9) {
        R.refused = true;
        R.refusal = "REFUSED (N, n) = (" + std::to_string(N) + ", " + std::to_string(n) + "): the " +
                    (tree_pair ? std::string("tree cusp pair (D5c)") : std::string("dense pair")) + " piece predicts " +
                    std::to_string(R.rss_pred >> 20) + " MiB above 0.9 x the ceiling " + std::to_string(opt.rss_cap_bytes >> 20) +
                    " MiB (D = " + std::to_string(R.D) + "; refused before building, rule 19)";
        R.t_total = since(t0);
        return R;
    }
    const uint32_t ellp = opt.rank_prime ? opt.rank_prime : DefaultRankPrime(n);
    std::vector<PairLocal> locals;
    std::vector<uint32_t> ran;
    // (PAR-ASM (c)) the pieces, each a function of (ell) alone: its report, its local LES and its verdict.  The ell <= 3 pieces (the
    // large, individually predicted ones) run first, one after another, each with its own RSS sample as before; the ell >= 5 pieces
    // are independent and run concurrently under the joint RSS prediction (rule 19: at most floor(0.9 cap / one piece) at a time,
    // one piece predicted as the dense 3-adic pair's PredictPairRss(3, D, T), whose cochains bound the Bredon / E8 model's); their
    // reports then carry the joint RSS sample.  Every result lands in its own slot and is consumed below in the serial (prime)
    // order, so the groups, maps, refusal text and counts are those of the serial loop.  Lever HECKE_DESCENT_PAR_ELL=0: serial.
    struct PieceOut {
        PieceReport rep;
        PairLocal L;
        bool ok = true;
    };
    std::vector<uint32_t> todo;
    for (uint32_t ell : R.primes)
        if (opt.only_primes.empty() || std::find(opt.only_primes.begin(), opt.only_primes.end(), ell) != opt.only_primes.end())
            todo.push_back(ell);
    std::vector<PieceOut> outs(todo.size());
    auto run_piece = [&](uint32_t ell, bool sample, PieceOut& out) {
        PieceReport& rep = out.rep;
        PairLocal& L = out.L;
        bool& ok = out.ok;
        rep.ell = ell;
        rep.min_prec = 1000;
        rep.T = T;
        const auto tp = clk::now();
#if defined(__GLIBC__)
        if (sample) malloc_trim(0);
#endif
        if (sample) rep.rss_base = CurrentRssBytes();
        std::unique_ptr<RssSampler> rs;
        if (sample) rs = std::make_unique<RssSampler>();
        ok = true;
        {
            const uint32_t K = ell == 2 ? (opt.test_K_two ? opt.test_K_two : LocalWidth(2))
                                        : (ell == 3 ? LocalWidth(3) : integral_model::DefaultKBudget(ell));
            rep.K = K;
            const U32Ctx ctx = U32Ctx::prime_power(ell, K);
            const auto tb = clk::now();
            PairComplex X;
            const LocalModel* model = nullptr;
            const LocalModel* rank_model = nullptr;   // the O_ell-model whose cone differentials are ranked mod ell'
            bool free_model = false;
            bool tree_piece = false;
            if (ell <= 3 && tree_pair) {
                // D5c (sec:treepair): the tree cone Cone(a^(ell)*) in closed form (prop:treepair(2)); its certificates (d_B d_B = 0,
                // a^* a cochain map) throw hecke::HeckeInternalError, never a refusal (prop:treepair(6))
                rep.rss_pred = treepair::PredictTreePairRss(ell, R.D, T);
                X = treepair::BuildTreePairComplex(ell, ctx, integral_model::TreeInducedSymModule(N, n, ctx), T);
                tree_piece = true;
            } else if (ell <= 3) {
                if (ell == 2) rep.rss_pred = PredictPairRss(2, R.D, T);
                else rep.rss_pred = PredictPairRss(3, R.D, T);
                const CoefficientModule<uint32_t> M = InducedSymModule(N, n, ctx);
                if (opt.route == PairRoute::IsoFree) {
                    X = BuildFreePair(ctx, M, T);
                    model = &FreeModel();
                    free_model = true;
                } else {
                    model = ell == 2 ? &Iso2() : &Iso3();
                    X = BuildModelPair(*model, ctx, M, T);
                }
            } else if (N == 1 && opt.level1_e8) {
                // audit m3: the identity of complexes (prop:descentpair(1)(d)) is certified INSIDE the assembly, every call: eps^*
                // intertwines the shipped coboundaries and restriction with the Bredon model's and is onto its cochains; the ell'
                // rank source then runs on that Bredon model (the same cochains by (1)(d))
                E8Identity id;
                X = BuildE8Pair(n, ell, K, T, &id);
                ++rep.checks;
                if (!id.all()) {
                    ok = false;
                    rep.why = "the shipped E8 pair is not identified with the Bredon pair (prop:descentpair(1)(d)):" + id.why;
                }
                rank_model = &BredonModel(ell);
            } else {
                model = &BredonModel(ell);
                X = BuildModelPair(*model, ctx, InducedSymModule(N, n, ctx), T);
            }
            if (model) rank_model = model;
            rep.build_seconds = since(tb);
            // (NBCONFIRM S5) between the piece's build and its reads; only on the serial calls (sample = true: the caller's thread,
            // outside any parallel region), never in the ForEachIndex body below
            if (sample) oc_util::CancelPoint("descent.pair_piece");
            rep.route = X.route;
            rep.c = X.c;
            rep.cb = X.cb;
            rep.matrix_bytes = X.bytes();
            // code map (a) at ell = 3 (and the Bredon model at ell >= 5): the boundary is the circle model -- every boundary
            // column non-periodic (no boundary cochains above degree 1) and its coboundary the t = 0 block of the free model's
            if (model && !free_model && ell >= 3) {
                for (uint32_t q = 2; q <= T; ++q)
                    if (X.cb[q] != 0) {
                        ok = false;
                        rep.why = "the ell = " + std::to_string(ell) + " boundary has cochains in degree " + std::to_string(q) +
                                  " (not the circle model, prop:descentpair(1)(d))";
                    }
                const CoefficientModule<uint32_t> M = InducedSymModule(N, n, ctx);
                const Mat Lb = FromDenseBuf(model->boundary_coboundary(ctx, M, 0));
                const Mat Fb = FromDenseBuf(FreeModel().boundary_coboundary(ctx, M, 0));
                const Mat P0 = FromDenseBuf(model->boundary_projector(ctx, M, 0));
                // the free model's degree-1 boundary cells: the t = 0 edges first (s = 0 cells come first in the cell order, so
                // pick the s = 1 rows explicitly)
                std::vector<uint32_t> rows;
                {
                    uint32_t b = 0;
                    for (const auto& cl : FreeModel().cells(1)) {
                        if (!cl.boundary) continue;
                        if (cl.s == 1)
                            for (uint32_t a = 0; a < M.dim; ++a) rows.push_back(b * M.dim + a);
                        ++b;
                    }
                }
                std::vector<uint32_t> cols(Fb.c);
                for (uint32_t i = 0; i < Fb.c; ++i) cols[i] = i;
                const Mat Fsub = Sub(Fb, rows, cols);
                if (ok && !EqualMat(X.Z, MatMulZ(X.Z, Fsub, P0), Lb)) {
                    ok = false;
                    rep.why = "the ell = " + std::to_string(ell) + " boundary coboundary is not the circle model's (t = 0 block of Iso)";
                }
                ++rep.checks;
            }
            // code map (a) at ell = 2: the boundary is FREE, Isob^(2)_q = O_2[Gamma]^{w_q}, w = 3, 6, 6, ... (every boundary stabiliser
            // is its own 2-Sylow, Q_sigma = 1: the reduced boundary cochains are the full ones, D per cell), non-zero in every degree
            if (ell == 2 && model && !free_model) {
                for (uint32_t q = 0; q <= T; ++q)
                    if (X.cb[q] != (q == 0 ? 3u : 6u) * R.D) {
                        ok = false;
                        rep.why = "the ell = 2 boundary is not free of rank 3, 6, 6, ... (prop:descentpair(1)(d))";
                    }
                ++rep.checks;
            }
            // D5c: the cusp model's ranks (def:treepair(1)): free of rank 1, 2, 2, ... at 2; M^{-I} in degrees 0, 1 at 3, i.e.
            // rank(e_E M) = D for n even and 0 for n odd (rho(-I) = (-1)^n on M_N; reached for n odd only through extra_primes = {3}:
            // the D5c audit's m3 -- the builder's selection rank rE must be this value)
            if (tree_piece) {
                const uint32_t rE = n % 2 == 0 ? R.D : 0u;
                for (uint32_t q = 0; q <= T; ++q) {
                    const uint32_t want = ell == 2 ? (q == 0 ? 1u : 2u) * R.D : (q <= 1 ? rE : 0u);
                    if (X.cb[q] != want) {
                        ok = false;
                        rep.why = "the cusp model's cochain ranks are not those of def:treepair(1) at ell = " + std::to_string(ell);
                    }
                }
                ++rep.checks;
            }
            if (ok && opt.test_break_res) {
                for (uint32_t q = 0; q <= T; ++q)
                    if (X.r[q].r && X.r[q].c) {
                        X.r[q].at(0, 0) = X.Z.add(X.r[q].at(0, 0), 1);
                        break;
                    }
            }
            // code map (c): the cone differentials mod ell' on the SAME O_ell-model (full layout)
            if (ok && opt.prime_rank_check && (rank_model || tree_piece)) {
                const auto tr = clk::now();
                const bool rule = ellp >= 5 && IsPrime(ellp) && ellp > n + 1 && QEllOf(ell) % ellp != 0 && ellp != ell;
                if (!rule && !opt.test_skip_prime_rule) {
                    ok = false;
                    rep.why = "the rank prime ell' = " + std::to_string(ellp) +
                              " does not qualify (ell' >= 5, ell' > n + 1, prime to q_ell; code map (c): never ell' in {2, 3})";
                } else {
                    rep.rank_prime = ellp;
                    // D5c: on THIS tree cone (prop:treepair(5)(a): never a rank of (Iso, Isob)'s cone)
                    const std::vector<uint32_t> rp = tree_piece ? treepair::TreeConeRanksModPrime(ell, N, n, ellp, T)
                                                                : ConeRanksModPrime(*rank_model, N, n, ellp, T, free_model);
                    // the recursion
                    int64_t pc = 0;
                    for (int j = -1; j + 2 <= static_cast<int>(T); ++j) {
                        const int64_t x = int64_t(ConeRank(X, j)) - int64_t(R.bt.bc(j + 1)) - pc;
                        pc = x;
                        if (x != int64_t(rp[static_cast<size_t>(j + 1)])) {
                            ok = false;
                            rep.why = "rank of D^" + std::to_string(j) + " mod ell' = " + std::to_string(ellp) + " is " +
                                      std::to_string(rp[static_cast<size_t>(j + 1)]) + ", the Betti recursion gives " + std::to_string(x) +
                                      " (ell' is not a rank source here; prop:descentpair(6)(c))";
                            break;
                        }
                        ++rep.rank_checks;
                    }
                }
                rep.rank_seconds = since(tr);
            }
            if (ok) ok = RunPairLes(X, R.bt, opt.q_max, L, rep, opt.test_rank_offset);
            else rep.ok = false;
            // code map (d): the boundary against the closed form from the cusp widths
            if (ok) {
                for (uint32_t q = 0; q <= opt.q_max; ++q) {
                    bool cok = false;
                    const AbelianGroupType cf = BoundaryClosedForm(R.widths, n, ell, q, K, cok);
                    const AbelianGroupType got = PrimaryWithFree(L.Hd[q].type(), ell);
                    if (!cok || cf != got) {
                        ok = false;
                        rep.ok = false;
                        rep.why = "H^" + std::to_string(q) + "(d) = " + got.to_string() + " at ell = " + std::to_string(ell) +
                                  " differs from the closed form " + cf.to_string() + " (widths)";
                        break;
                    }
                    ++rep.checks;
                }
                if (ok && even) {   // |tors H^1(d)|_ell = prod_c n! w_c^n (prop:descentpair(4), (6))
                    uint32_t v = 0, got = 0;
                    for (uint32_t w : R.widths) v += ValuationFactorialPower(n, w, ell);
                    for (uint32_t a : L.Hd[1].tors) got += a;
                    if (v != got) {
                        ok = false;
                        rep.ok = false;
                        rep.why = "v_ell |tors H^1(d)| = " + std::to_string(got) + " != sum_c v_ell(n! w_c^n) = " + std::to_string(v);
                    }
                    ++rep.checks;
                }
            }
            // (5)(e) at ell >= 5: tors H^2_c (x) Z_(ell) = tors H_0(Gamma; M) (x) Z_(ell), H_0 = coker [rho(S) - 1 | rho(T) - 1]
            if (ok && ell >= 5) {
                const CoefficientModule<uint32_t> M = InducedSymModule(N, n, ctx);
                const DenseBuf<uint32_t> RS = M.rho(Mat22::S()), RT = M.rho(Mat22::T());
                const uint32_t Dm = M.dim;
                Mat A(Dm, 2 * Dm);
                for (uint32_t i = 0; i < Dm; ++i)
                    for (uint32_t k = 0; k < Dm; ++k) {
                        A.at(i, k) = X.Z.sub(RS.at(i, k), i == k ? 1u : 0u);
                        A.at(i, Dm + k) = X.Z.sub(RT.at(i, k), i == k ? 1u : 0u);
                    }
                const maps::Smith Sm = maps::SmithForm(X.Z, A, false, false);
                std::vector<uint32_t> h0, hc2 = L.Hc[2].tors;
                for (uint32_t s : Sm.sigma)
                    if (s >= 1) h0.push_back(s);
                std::sort(h0.begin(), h0.end());
                std::sort(hc2.begin(), hc2.end());
                if (Sm.rank() != Dm - R.bt.b0 || h0 != hc2) {
                    ok = false;
                    rep.ok = false;
                    rep.why = "tors H^2_c (x) Z_(ell) differs from tors H_0(Gamma; M) (x) Z_(ell) (prop:descentpair(5)(e))";
                }
                if (ok && ell > n + 1 && !hc2.empty()) {
                    ok = false;
                    rep.ok = false;
                    rep.why = "ell > n + 1 divides tors H^2_c (prop:descentpair(5)(e))";
                }
                ++rep.checks;
            }
            // an extra prime outside S: every term torsion-free
            if (ok && extra.count(ell)) {
                for (uint32_t q = 0; q <= opt.q_max; ++q)
                    if (!L.H[q].tors.empty() || !L.Hd[q].tors.empty() || !L.Hc[q].tors.empty()) {
                        ok = false;
                        rep.ok = false;
                        rep.why = "a prime outside S carries torsion at q = " + std::to_string(q);
                    }
                ++rep.checks;
            }
        }
        if (rs) rep.rss_meas = rs->stop();
        rep.seconds = since(tp);
    };
    size_t first5 = 0;
    while (first5 < todo.size() && todo[first5] <= 3) ++first5;
    for (size_t i = 0; i < first5; ++i) {
        oc_util::CancelPoint("descent.pair_piece");   // (NBCONFIRM S5) per ell <= 3 piece (outs is stack-local)
        run_piece(todo[i], true, outs[i]);
    }
    {
        const uint32_t m5 = static_cast<uint32_t>(todo.size() - first5);
        uint32_t w5 = 1;
        R.ell_rss_pred_one = PredictPairRss(3, R.D, T);
        if (m5 > 1 && par::Enabled(par::Lever::Ell) && !par::DeviceRequested()) {
            const uint32_t th = par::Threads();
            w5 = par::JointWorkers(m5, R.ell_rss_pred_one, opt.rss_cap_bytes, th);
            par::NoteEll(w5 > 1, w5 < std::min(th, m5));
        }
        R.ell_workers = w5;
        R.ell_rss_pred_joint = R.ell_rss_pred_one * w5;
        // (audit F6, as cdefc684 did for level 1) the whole call's prediction counts the ell >= 5 stage: the ell <= 3 pieces ran first and
        // freed their matrices, so the call's peak is the larger of the two stages.  JointWorkers keeps the joint term below 0.9 x the
        // ceiling, so no new refusal can arise; the figure is what an operator applying rule 19 to rss_pred must see.
        if (m5 > 0) R.rss_pred = std::max(R.rss_pred, R.ell_rss_pred_joint);
        if (w5 <= 1) {
            for (size_t i = first5; i < todo.size(); ++i) {
                oc_util::CancelPoint("descent.pair_piece");   // (NBCONFIRM S5) per ell >= 5 piece, serial route
                run_piece(todo[i], true, outs[i]);
            }
        } else {
            oc_util::CancelPoint("descent.pair_piece");   // (NBCONFIRM S5) before the parallel region only, never inside it
#if defined(__GLIBC__)
            malloc_trim(0);
#endif
            const uint64_t base = CurrentRssBytes();
            RssSampler rs;
            par::ForEachIndex(m5, w5, [&](size_t i) { run_piece(todo[first5 + i], false, outs[first5 + i]); });
            const uint64_t peak = rs.stop();
            for (size_t i = first5; i < todo.size(); ++i) {   // the joint sample (the pieces ran together)
                outs[i].rep.rss_base = base;
                outs[i].rep.rss_meas = peak;
            }
            R.ell_rss_growth = peak > base ? peak - base : 0;
        }
    }
    for (size_t i = 0; i < todo.size(); ++i) {
        const uint32_t ell = todo[i];
        PieceReport& rep = outs[i].rep;
        if (!outs[i].ok) why << " ell=" << ell << " (" << rep.route << "): " << rep.why << ";";
        R.exact_spots += rep.exact_spots;
        R.checks += rep.checks + rep.rank_checks;
        R.pieces.push_back(rep);
        if (outs[i].ok) {
            locals.push_back(std::move(outs[i].L));
            ran.push_back(ell);
        }
    }
    if (!opt.only_primes.empty()) why << " only_primes set (a measurement, not a result);";
    // audit m4: code map (c) certifies D^q, q >= 0, by ell' as well; with the ell' check off the result is a measurement only
    if (!opt.prime_rank_check) why << " prime_rank_check off (the ell' rank source of code map (c) not run: a measurement, not a result);";
    if (why.str().empty()) {
        const uint32_t qm = opt.q_max;
        auto glue = [&](auto getter, uint32_t q, AbelianGroupType& out, const char* what) {
            bool first = true;
            for (const PairLocal& L : locals) {
                const AbelianGroupType g = getter(L, q);
                if (first) out.free_rank = g.free_rank;
                else if (g.free_rank != out.free_rank) why << " free rank of " << what << " at q=" << q << " differs between the pieces;";
                first = false;
                for (const auto& [p, ex] : g.torsion)
                    for (uint32_t a : ex) out.add_torsion(p, a);
            }
        };
        R.H.resize(qm + 1);
        R.Hd.resize(qm + 1);
        R.Hc.resize(qm + 1);
        R.res.resize(qm + 1);
        R.proj.resize(qm + 1);
        R.j.resize(qm);
        for (uint32_t q = 0; q <= qm; ++q) {
            glue([](const PairLocal& L, uint32_t x) { return L.H[x].type(); }, q, R.H[q], "H");
            glue([](const PairLocal& L, uint32_t x) { return L.Hd[x].type(); }, q, R.Hd[q], "H(d)");
            glue([](const PairLocal& L, uint32_t x) { return L.Hc[x].type(); }, q, R.Hc[q], "H_c");
            glue([](const PairLocal& L, uint32_t x) { return L.res[x].ker; }, q, R.res[q].ker, "ker res");
            glue([](const PairLocal& L, uint32_t x) { return L.res[x].im; }, q, R.res[q].im, "im res");
            glue([](const PairLocal& L, uint32_t x) { return L.res[x].coker; }, q, R.res[q].coker, "coker res");
            glue([](const PairLocal& L, uint32_t x) { return L.proj[x].ker; }, q, R.proj[q].ker, "ker proj");
            glue([](const PairLocal& L, uint32_t x) { return L.proj[x].im; }, q, R.proj[q].im, "im proj");
            glue([](const PairLocal& L, uint32_t x) { return L.proj[x].coker; }, q, R.proj[q].coker, "coker proj");
            if (q < qm) {
                glue([](const PairLocal& L, uint32_t x) { return L.j[x].ker; }, q, R.j[q].ker, "ker iota");
                glue([](const PairLocal& L, uint32_t x) { return L.j[x].im; }, q, R.j[q].im, "im iota");
                glue([](const PairLocal& L, uint32_t x) { return L.j[x].coker; }, q, R.j[q].coker, "coker iota");
            }
            std::ostringstream cf;
            cf << "H^" << q << "(d) = " << R.Hd[q].to_string() << " (widths";
            for (uint32_t w : R.widths) cf << " " << w;
            cf << ")";
            R.closed_form.push_back(cf.str());
        }
    }
    R.t_total = since(t0);
    const std::string w = why.str();
    if (!w.empty()) {
        R.refused = true;
        R.refusal = "REFUSED (N, n) = (" + std::to_string(N) + ", " + std::to_string(n) + "):" + w;
        R.H.clear();
        R.Hd.clear();
        R.Hc.clear();
        R.res.clear();
        R.j.clear();
        R.proj.clear();
    }
    return R;
}

}  // namespace pair
}  // namespace integral_descent
