// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// DESCENT milestone D2: restriction, corestriction, the level cone and its long exact sequence over Z.  See the header;
// [C] = docs/latex/gl2_contract.tex (v1.23.0-pre.22; the precision rule is prop:descentmaps(2)(g)), cited by label.
#include "DescentMaps.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#if defined(__GLIBC__)
#include <malloc.h>   // malloc_trim
#endif
#include <map>
#include <numeric>
#include <set>
#include <sstream>

#include "DescentLevelN.h"
#include "DescentSmith.h"
#include "DescentParAsm.h"
#include "DescentTree.h"
#include "../Model/TreeCochains.h"
#include "../../Hecke/HeckeLevelN.h"
#include "../../Hecke/HeckeLevelPair.h"
#include "../../../ModArithmetic/EngineCheck.h"

namespace integral_descent {
namespace maps {

using engine::DenseBuf;
using U32Ctx = engine::ModCtx<uint32_t>;
using clk = std::chrono::steady_clock;
namespace les = hecke::level::les;

namespace {
double since(clk::time_point t0) { return std::chrono::duration<double>(clk::now() - t0).count(); }
}  // namespace

// ---------------------------------------------------------------------------
// Z/ell^k
// ---------------------------------------------------------------------------
ZMod::ZMod(uint32_t ell_, uint32_t k_) : ell(ell_), k(k_) {
    ENGINE_CHECK_MSG(ell >= 2 && IsPrime(ell) && k >= 1, "ZMod: ell prime, k >= 1");
    uint64_t v = 1;
    for (uint32_t i = 0; i < k && v < (uint64_t(1) << 31); ++i) v *= ell;
    ENGINE_CHECK_MSG(v < (uint64_t(1) << 31), "ZMod: ell^k must stay below 2^31");
    ctx = U32Ctx::chain(ell, k);
    m = ctx.p;
    two = ell == 2;
    mask = two ? static_cast<uint32_t>(m - 1) : 0u;
}
uint32_t ZMod::val(uint32_t x) const {
    x = red(x);
    if (two && x) return static_cast<uint32_t>(__builtin_ctz(x));
    return ctx.valuation(x);
}
uint32_t ZMod::inv(uint32_t u) const {
    u = red(u);
    ENGINE_CHECK_MSG(u % ell != 0, "ZMod::inv: not a unit");
    return ctx.inv(u);
}
uint32_t ZMod::pw(uint32_t v) const {
    ENGINE_CHECK(v <= k);
    return v == k ? 0u : ctx.ell_pow(v);
}

// ---------------------------------------------------------------------------
// matrices
// ---------------------------------------------------------------------------
namespace {

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
// A (r x n) * B (n x c), entries already reduced mod Z.m.
// PAR-D2: on the lever, the threaded SIMD product (DescentSmith.h MulMod; the same exact values).
Mat Mul(const ZMod& Z, const Mat& A, const Mat& B) {
    ENGINE_CHECK_MSG(A.c == B.r, "maps::Mul: shape");
    if (DescentSmithBlockedEnabled()) return MulMod(Z, A, B);
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
            const uint64_t x = a[k];
            if (!x) continue;
            const uint32_t* b = B.row(k);
            for (uint32_t j = 0; j < B.c; ++j) acc[j] += x * b[j];
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
Mat FromDense(const DenseBuf<uint32_t>& D) {
    Mat A(D.rows, D.cols);
    for (uint32_t i = 0; i < D.rows; ++i)
        for (uint32_t j = 0; j < D.cols; ++j) A.at(i, j) = D.at(i, j);
    return A;
}
Mat FromRowMajor(uint32_t r, uint32_t c, const std::vector<uint32_t>& v) {
    ENGINE_CHECK(v.size() == size_t(r) * c);
    Mat A(r, c);
    A.a = v;
    return A;
}
Mat Rows(const Mat& A, uint32_t from, uint32_t to) {
    Mat B(to - from, A.c);
    for (uint32_t i = from; i < to; ++i) std::copy(A.row(i), A.row(i) + A.c, B.row(i - from));
    return B;
}
// a := a - f b (length n)
inline void RowSub(const ZMod& Z, uint32_t* a, const uint32_t* b, uint32_t f, uint32_t n) {
    if (Z.two) {
        for (uint32_t j = 0; j < n; ++j) a[j] = (a[j] - f * b[j]) & Z.mask;
    } else {
        for (uint32_t j = 0; j < n; ++j) {
            if (!b[j]) continue;
            const uint32_t t = static_cast<uint32_t>(uint64_t(f) * b[j] % Z.m);
            a[j] = a[j] >= t ? a[j] - t : static_cast<uint32_t>(a[j] + Z.m - t);
        }
    }
}
inline void RowAdd(const ZMod& Z, uint32_t* a, const uint32_t* b, uint32_t f, uint32_t n) {
    RowSub(Z, a, b, static_cast<uint32_t>((Z.m - f) % Z.m), n);
}
inline uint32_t DivPow(const ZMod& Z, uint32_t x, uint32_t v) {
    if (Z.two) return x >> v;
    uint32_t y = x;
    for (uint32_t i = 0; i < v; ++i) y /= Z.ell;
    return y;
}
// Inverse of a matrix invertible mod ell (unit pivots), Gauss-Jordan mod ell^k.
Mat InverseUnit(const ZMod& Z, Mat G) {
    ENGINE_CHECK(G.r == G.c);
    const uint32_t n = G.r;
    Mat I = Identity(n);
    for (uint32_t t = 0; t < n; ++t) {
        uint32_t p = t;
        while (p < n && Z.val(G.at(p, t)) != 0) ++p;
        ENGINE_CHECK_MSG(p < n, "maps::InverseUnit: the block P[R, J] is not invertible mod ell");
        if (p != t) {
            for (uint32_t j = 0; j < n; ++j) {
                std::swap(G.at(t, j), G.at(p, j));
                std::swap(I.at(t, j), I.at(p, j));
            }
        }
        const uint32_t u = Z.inv(G.at(t, t));
        for (uint32_t j = 0; j < n; ++j) {
            G.at(t, j) = Z.mul(G.at(t, j), u);
            I.at(t, j) = Z.mul(I.at(t, j), u);
        }
        for (uint32_t i = 0; i < n; ++i) {
            if (i == t || !G.at(i, t)) continue;
            const uint32_t f = G.at(i, t);
            RowSub(Z, G.row(i), G.row(t), f, n);
            RowSub(Z, I.row(i), I.row(t), f, n);
        }
    }
    return I;
}
void AddTorsion(AbelianGroupType& G, uint32_t ell, const std::vector<uint32_t>& sigma) {
    for (uint32_t s : sigma)
        if (s >= 1) G.add_torsion(ell, s);
}
// (PAR-ASM audit F1) atomic: FactorCounted runs from the concurrent ell >= 5 pair pieces (DescentPair.cpp) -- a counter, no value
std::atomic<uint32_t> g_smith_forms{0};

// PAR-D2: the factored Smith form (DescentSmith.h), counted and captured as maps::SmithForm is.
SmithFactors FactorCounted(const ZMod& Z, const Mat& A, bool want_g) {
    ++g_smith_forms;
    SmithCaptureNote(Z, A);
    return SmithFactor(Z, A, want_g);
}
SmithFactors FactorCounted(const ZMod& Z, Mat&& A, bool want_g) {
    ++g_smith_forms;
    SmithCaptureNote(Z, A);
    return SmithFactor(Z, std::move(A), want_g);
}

}  // namespace

// ---------------------------------------------------------------------------
// Smith form with transforms (minimal-valuation full pivoting)
// ---------------------------------------------------------------------------
uint32_t Smith::smax() const {
    uint32_t s = 0;
    for (uint32_t v : sigma) s = std::max(s, v);
    return s;
}

Smith SmithForm(const ZMod& Z, Mat A, bool left, bool right) {
    ++g_smith_forms;
    SmithCaptureNote(Z, A);
    if (DescentSmithBlockedEnabled()) return SmithFormBlocked(Z, std::move(A), left, right);
    return SmithFormOracle(Z, std::move(A), left, right);
}

// The oracle of record (D2's scalar code, unchanged; HECKE_DESCENT_SMITH_BLOCKED=0 routes every call here).
Smith SmithFormOracle(const ZMod& Z, Mat A, bool left, bool right) {
    Smith S;
    const uint32_t r = A.r, c = A.c;
    if (left) {
        S.L = Identity(r);
        S.LinvT = Identity(r);
    }
    if (right) {
        S.RT = Identity(c);
        S.Rinv = Identity(c);
    }
    for (auto& x : A.a) x = Z.red(x);
    uint32_t t = 0;
    while (t < r && t < c) {
        uint32_t best = Z.k, bi = 0, bj = 0;
        bool unit = false;
        for (uint32_t i = t; i < r && !unit; ++i) {
            const uint32_t* row = A.row(i);
            for (uint32_t j = t; j < c; ++j) {
                const uint32_t x = row[j];
                if (!x) continue;
                if (Z.two ? (x & 1u) : (x % Z.ell != 0)) {
                    best = 0;
                    bi = i;
                    bj = j;
                    unit = true;
                    break;
                }
                const uint32_t v = Z.val(x);
                if (v < best) {
                    best = v;
                    bi = i;
                    bj = j;
                }
            }
        }
        if (best >= Z.k) break;
        if (bi != t) {
            std::swap_ranges(A.row(t), A.row(t) + c, A.row(bi));
            if (left) {
                std::swap_ranges(S.L.row(t), S.L.row(t) + r, S.L.row(bi));
                std::swap_ranges(S.LinvT.row(t), S.LinvT.row(t) + r, S.LinvT.row(bi));
            }
        }
        if (bj != t) {
            for (uint32_t i = t; i < r; ++i) std::swap(A.at(i, t), A.at(i, bj));
            if (right) {
                std::swap_ranges(S.RT.row(t), S.RT.row(t) + c, S.RT.row(bj));
                std::swap_ranges(S.Rinv.row(t), S.Rinv.row(t) + c, S.Rinv.row(bj));
            }
        }
        const uint32_t v = best;
        const uint32_t uinv = Z.inv(DivPow(Z, A.at(t, t), v));
        uint32_t* prow = A.row(t);
        for (uint32_t i = t + 1; i < r; ++i) {
            const uint32_t x = A.at(i, t);
            if (!x) continue;
            const uint32_t f = Z.mul(DivPow(Z, x, v), uinv);
            RowSub(Z, A.row(i) + t, prow + t, f, c - t);
            ENGINE_DEBUG_CHECK(A.at(i, t) == 0);
            A.at(i, t) = 0;
            if (left) {
                RowSub(Z, S.L.row(i), S.L.row(t), f, r);
                RowAdd(Z, S.LinvT.row(t), S.LinvT.row(i), f, r);
            }
        }
        for (uint32_t j = t + 1; j < c; ++j) {
            const uint32_t x = prow[j];
            if (!x) continue;
            prow[j] = 0;
            if (right) {
                const uint32_t g = Z.mul(DivPow(Z, x, v), uinv);
                RowSub(Z, S.RT.row(j), S.RT.row(t), g, c);
                RowAdd(Z, S.Rinv.row(t), S.Rinv.row(j), g, c);
            }
        }
        S.sigma.push_back(v);
        ++t;
    }
    return S;
}

// ---------------------------------------------------------------------------
// presentations
// ---------------------------------------------------------------------------
AbelianGroupType Pres::type() const {
    AbelianGroupType G;
    G.free_rank = free;
    AddTorsion(G, Z.ell, tors);
    return G;
}

Pres ZeroPres(const ZMod& Z, uint32_t c) {
    Pres H;
    H.Z = Z;
    H.c = c;
    H.gensT = Mat(0, c);
    H.coord = Mat(0, c);
    return H;
}

namespace {
Pres Refuse(Pres H, const std::string& why) {
    H.ok = false;
    H.why = why;
    return H;
}
}  // namespace

namespace {
// The presentation code of record (D2), on the oracle's dense transforms (HECKE_DESCENT_SMITH_BLOCKED=0).
Pres BuildPresLegacy(const ZMod& Z, const Mat& P, uint32_t rankP, const Mat& Nx, uint32_t c, uint32_t b) {
    Pres H = ZeroPres(Z, c);
    ENGINE_CHECK_MSG(P.r == c, "BuildPres: the previous differential must land in the cochains");
    if (c == 0) {
        ENGINE_CHECK(rankP == 0 && b == 0);
        return H;
    }
    Smith S1 = SmithForm(Z, P, true, false);
    const uint32_t s = S1.rank();
    // r_K > rank_Q is impossible for a complex with the stated Betti numbers ([C, prop:certifiede](1)): a wrong rank input or a
    // cochain "map" that is not one (the negative control) -- refused, never thrown past the caller.
    if (s > rankP)
        return Refuse(H, "r_K(d^{q-1}) = " + std::to_string(s) + " > rank_Q = " + std::to_string(rankP) +
                             " (not a complex with these Betti numbers)");
    if (s < rankP)
        return Refuse(H, "d^{q-1} not certified mod " + std::to_string(Z.ell) + "^" + std::to_string(Z.k) + " (r_K = " +
                             std::to_string(s) + " < rank_Q = " + std::to_string(rankP) + ")");
    const uint32_t sm1 = S1.smax();
    if (Z.k <= 2 * sm1)
        return Refuse(H, "precision: the exponent " + std::to_string(sm1) + " of d^{q-1} leaves k - sigma_max <= sigma_max at k = " +
                             std::to_string(Z.k));
    const ZMod Z1(Z.ell, Z.k - sm1);
    std::vector<uint32_t> trows;
    for (uint32_t i = 0; i < s; ++i)
        if (S1.sigma[i] >= 1) {
            trows.push_back(i);
            H.tors.push_back(S1.sigma[i]);
        }
    ZMod ZF = Z1;
    Mat freeG, freeC;
    if (b > 0) {
        ENGINE_CHECK_MSG(Nx.c == c, "BuildPres: the next differential must start at the cochains");
        ENGINE_CHECK_MSG(c >= rankP + b, "BuildPres: Betti number too large");
        const uint32_t rankN = c - rankP - b;
        const Mat UrT = Reduce(Rows(S1.LinvT, s, c), Z1);   // (c - s) x c
        const Mat E = Mul(Z1, Reduce(Nx, Z1), Transpose(UrT));   // c' x (c - s)
        Smith S2 = SmithForm(Z1, E, false, true);
        if (S2.rank() > rankN) return Refuse(H, "r_K(d^q) > rank_Q (not a complex with these Betti numbers)");
        if (S2.rank() < rankN)
            return Refuse(H, "d^q on the complement not certified (r_K = " + std::to_string(S2.rank()) + " < " + std::to_string(rankN) + ")");
        const uint32_t sm2 = S2.smax();
        if (Z1.k <= sm2 + 1) return Refuse(H, "precision exhausted by d^q");
        ZF = ZMod(Z.ell, Z1.k - sm2);
        const Mat Kt = Reduce(Rows(S2.RT, rankN, c - s), ZF);   // b x (c - s)
        ENGINE_CHECK(Kt.r == b);
        freeG = Mul(ZF, Kt, Reduce(UrT, ZF));                   // b x c
        freeC = Mul(ZF, Reduce(Rows(S2.Rinv, rankN, c - s), ZF), Reduce(Rows(S1.L, s, c), ZF));   // b x c
    }
    for (uint32_t a : H.tors)
        if (a >= ZF.k) return Refuse(H, "precision: a torsion exponent reaches the precision left");
    const uint32_t t = static_cast<uint32_t>(trows.size());
    H.free = b;
    H.Z = ZF;
    H.gensT = Mat(t + b, c);
    H.coord = Mat(t + b, c);
    for (uint32_t i = 0; i < t; ++i) {
        for (uint32_t j = 0; j < c; ++j) {
            H.gensT.at(i, j) = ZF.red(S1.LinvT.at(trows[i], j));
            H.coord.at(i, j) = ZF.red(S1.L.at(trows[i], j));
        }
    }
    for (uint32_t i = 0; i < b; ++i)
        for (uint32_t j = 0; j < c; ++j) {
            H.gensT.at(t + i, j) = freeG.at(i, j);
            H.coord.at(t + i, j) = freeC.at(i, j);
        }
    return H;
}
}  // namespace

// PAR-D2: the same presentation from the FACTORED Smith forms, reading only what it uses (DescentSmith.h, PARD2_report.md Sec. 3):
// the rows >= s of L^{-1}T are unit vectors e_{rowOrig[t]}, so U_r^T is a column selection (E = the columns rowOrig[s..c) of Nx) and
// a scatter (freeG); the rows >= rank of R2^{-1} are unit vectors e_{colOrig2[t]}, so freeC is b rows of L; the kernel basis is the
// rows >= rank of R2^T (a triangular solve); only the torsion rows and these b rows of L are formed.  Entry for entry the legacy
// values (every step is the same exact ring map); the order of the refusals is the legacy order.
Pres BuildPres(const ZMod& Z, const Mat& P, uint32_t rankP, const Mat& Nx, uint32_t c, uint32_t b) {
    if (!DescentSmithBlockedEnabled()) {
        CallerCaptureBuildPres(Z, P, rankP, Nx, c, b);
        return BuildPresLegacy(Z, P, rankP, Nx, c, b);
    }
    return BuildPres(Z, Mat(P), rankP, Nx, c, b);
}
// The consuming form: the factorisation runs in P's own storage (one c x a matrix less at the peak).
Pres BuildPres(const ZMod& Z, Mat&& P, uint32_t rankP, const Mat& Nx, uint32_t c, uint32_t b) {
    CallerCaptureBuildPres(Z, P, rankP, Nx, c, b);
    if (!DescentSmithBlockedEnabled()) return BuildPresLegacy(Z, P, rankP, Nx, c, b);
    Pres H = ZeroPres(Z, c);
    ENGINE_CHECK_MSG(P.r == c, "BuildPres: the previous differential must land in the cochains");
    if (c == 0) {
        ENGINE_CHECK(rankP == 0 && b == 0);
        return H;
    }
    SmithFactors S1 = FactorCounted(Z, std::move(P), false);
    const uint32_t s = S1.rank();
    if (s > rankP)
        return Refuse(H, "r_K(d^{q-1}) = " + std::to_string(s) + " > rank_Q = " + std::to_string(rankP) +
                             " (not a complex with these Betti numbers)");
    if (s < rankP)
        return Refuse(H, "d^{q-1} not certified mod " + std::to_string(Z.ell) + "^" + std::to_string(Z.k) + " (r_K = " +
                             std::to_string(s) + " < rank_Q = " + std::to_string(rankP) + ")");
    const uint32_t sm1 = S1.smax();
    if (Z.k <= 2 * sm1)
        return Refuse(H, "precision: the exponent " + std::to_string(sm1) + " of d^{q-1} leaves k - sigma_max <= sigma_max at k = " +
                             std::to_string(Z.k));
    const ZMod Z1(Z.ell, Z.k - sm1);
    std::vector<uint32_t> trows;
    for (uint32_t i = 0; i < s; ++i)
        if (S1.sigma[i] >= 1) {
            trows.push_back(i);
            H.tors.push_back(S1.sigma[i]);
        }
    ZMod ZF = Z1;
    Mat Kt;                           // b x (c - s): the kernel basis of E (rows >= rank of R2^T)
    std::vector<uint32_t> lrows;      // the rows of L1 that freeC reads
    if (b > 0) {
        ENGINE_CHECK_MSG(Nx.c == c, "BuildPres: the next differential must start at the cochains");
        ENGINE_CHECK_MSG(c >= rankP + b, "BuildPres: Betti number too large");
        const uint32_t rankN = c - rankP - b;
        Mat E(Nx.r, c - s);   // = Nx U_r^T: the columns rowOrig[s ..) of Nx, mod ell^{k1}
        for (uint32_t i = 0; i < Nx.r; ++i) {
            const uint32_t* nx = Nx.row(i);
            uint32_t* e = E.row(i);
            for (uint32_t j = 0; j < c - s; ++j) e[j] = Z1.red(nx[S1.rowOrig[s + j]]);
        }
        SmithFactors S2 = FactorCounted(Z1, std::move(E), true);
        if (S2.rank() > rankN) return Refuse(H, "r_K(d^q) > rank_Q (not a complex with these Betti numbers)");
        if (S2.rank() < rankN)
            return Refuse(H, "d^q on the complement not certified (r_K = " + std::to_string(S2.rank()) + " < " + std::to_string(rankN) + ")");
        const uint32_t sm2 = S2.smax();
        if (Z1.k <= sm2 + 1) return Refuse(H, "precision exhausted by d^q");
        ZF = ZMod(Z.ell, Z1.k - sm2);
        std::vector<uint32_t> krows(c - s - rankN);
        for (uint32_t t = 0; t < krows.size(); ++t) krows[t] = rankN + t;
        Kt = FactorsRTRows(S2, krows);
        for (auto& x : Kt.a) x = ZF.red(x);
        ENGINE_CHECK(Kt.r == b);
        for (uint32_t i = 0; i < b; ++i) lrows.push_back(s + S2.colOrig[rankN + i]);
    }
    for (uint32_t a : H.tors)
        if (a >= ZF.k) return Refuse(H, "precision: a torsion exponent reaches the precision left");
    const uint32_t t = static_cast<uint32_t>(trows.size());
    H.free = b;
    H.Z = ZF;
    H.gensT = Mat(t + b, c);
    H.coord = Mat(t + b, c);
    std::vector<uint32_t> need = trows;
    need.insert(need.end(), lrows.begin(), lrows.end());
    const Mat Lr = FactorsLRows(S1, need);          // rows trows, then the b rows of freeC
    const Mat Gt = FactorsLinvTRows(S1, trows);
    for (uint32_t i = 0; i < t; ++i)
        for (uint32_t j = 0; j < c; ++j) {
            H.gensT.at(i, j) = ZF.red(Gt.at(i, j));
            H.coord.at(i, j) = ZF.red(Lr.at(i, j));
        }
    for (uint32_t i = 0; i < b; ++i) {
        uint32_t* g = H.gensT.row(t + i);
        for (uint32_t j = 0; j < c - s; ++j) g[S1.rowOrig[s + j]] = Kt.at(i, j);
        for (uint32_t j = 0; j < c; ++j) H.coord.at(t + i, j) = ZF.red(Lr.at(t + i, j));
    }
    return H;
}

// ---------------------------------------------------------------------------
// lattices and maps
// ---------------------------------------------------------------------------
namespace {

struct LatQuot {
    bool ok = true;
    std::string why;
    bool precision = false;   // (ICENSUS-P1) !ok by a precision shortfall (MapRes::precision)
    AbelianGroupType LmodI, ZmodL;
};
// Z^g >= L (generators Lg, g x kL, rational rank rL) >= I (generators Ig, g x kI, rank rI).
LatQuot LatticeQuotient(const ZMod& Z, const Mat& Lg, uint32_t rL, const Mat& Ig, uint32_t rI) {
    LatQuot Q;
    const uint32_t g = Lg.r;
    ENGINE_CHECK(Ig.r == g);
    // PAR-D2: on the lever, the factored form and Y = L Ig by a forward solve (no L); else the oracle's dense L
    const bool blk = DescentSmithBlockedEnabled();
    Smith S;
    SmithFactors SF;
    if (blk)
        SF = FactorCounted(Z, Lg, false);
    else
        S = SmithForm(Z, Lg, true, false);
    const std::vector<uint32_t>& sig = blk ? SF.sigma : S.sigma;
    const uint32_t srank = static_cast<uint32_t>(sig.size());
    if (srank != rL) {
        Q.ok = false;
        Q.why = "lattice L not certified (" + std::to_string(srank) + " < " + std::to_string(rL) + ")";
        Q.precision = true;
        return Q;
    }
    Q.ZmodL.free_rank = g - rL;
    AddTorsion(Q.ZmodL, Z.ell, sig);
    const uint32_t lm = blk ? SF.smax() : S.smax();
    if (Z.k <= lm + 1) {
        Q.ok = false;
        Q.why = "precision exhausted by the lattice L";
        Q.precision = true;
        return Q;
    }
    const ZMod Zq(Z.ell, Z.k - lm);
    const Mat Y = blk ? FactorsApplyL(SF, Reduce(Ig, Z)) : Mul(Z, S.L, Reduce(Ig, Z));   // g x kI
    Mat C(rL, Ig.c);
    for (uint32_t i = 0; i < g; ++i)
        for (uint32_t j = 0; j < Ig.c; ++j) {
            const uint32_t y = Y.at(i, j);
            if (i >= rL) {
                if (y != 0) {
                    Q.ok = false;
                    Q.why = "I is not contained in L (a coordinate off the span)";
                    return Q;
                }
                continue;
            }
            if (y != 0 && Z.val(y) < sig[i]) {
                Q.ok = false;
                Q.why = "I is not contained in L (divisibility)";
                return Q;
            }
            C.at(i, j) = Zq.red(DivPow(Z, y, sig[i]));
        }
    Smith Sc = SmithForm(Zq, C, false, false);
    if (Sc.rank() != rI) {
        Q.ok = false;
        Q.why = "I in L not certified (" + std::to_string(Sc.rank()) + " < " + std::to_string(rI) + ")";
        Q.precision = true;
        return Q;
    }
    Q.LmodI.free_rank = rL - rI;
    AddTorsion(Q.LmodI, Z.ell, Sc.sigma);
    return Q;
}

ZMod MinZ(const ZMod& a, const ZMod& b) {
    ENGINE_CHECK_MSG(a.ell == b.ell, "maps: two precisions at different primes");
    return a.k <= b.k ? a : b;
}
// Delta of a presentation: g x t, column i = ell^{a_i} e_i.
Mat DeltaOf(const Pres& X, const ZMod& Z) {
    const uint32_t t = static_cast<uint32_t>(X.tors.size());
    Mat D(X.g(), t);
    for (uint32_t i = 0; i < t; ++i) D.at(i, i) = Z.pw(X.tors[i]);
    return D;
}

}  // namespace

namespace {
MapRes InducedMapTail(const Pres& X, const Pres& Y, MapRes R);
}  // namespace

MapRes InducedMap(const Pres& X, const Pres& Y, const Mat& f, uint32_t rankF) {
    CallerCaptureInducedMap(X, Y, f, rankF);
    MapRes R;
    R.rankF = rankF;
    ENGINE_CHECK_MSG(X.ok && Y.ok, "InducedMap: a presentation was refused");
    ENGINE_CHECK_MSG(f.r == Y.c && f.c == X.c, "InducedMap: the cochain map has the wrong shape");
    const ZMod Z = MinZ(X.Z, Y.Z);
    R.Z = Z;
    // (PAR-D2: on the lever the reductions of f and Y.coord are fused into the threaded product, no reduced copy)
    const bool blk = DescentSmithBlockedEnabled();
    const Mat T1 = blk ? MulMod(Z, f, Transpose(Reduce(X.gensT, Z)), true) : Mul(Z, Reduce(f, Z), Transpose(Reduce(X.gensT, Z)));   // Y.c x gX
    R.F = blk ? MulMod(Z, Y.coord, T1, true) : Mul(Z, Reduce(Y.coord, Z), T1);                                                     // gY x gX
    return InducedMapTail(X, Y, std::move(R));
}

// (PAR-ASM) f = the embedding e_{in_off + t} |-> e_{out_off + t}, t < w (Y.c x X.c, zero elsewhere).  Then T1 = f gensT^T has the
// rows out_off + t equal to row in_off + t of gensT^T and zeros elsewhere, so F = coord T1 = coord[:, out_off ..) (gensT[:, in_off ..))^T
// over the w middle indices only: the same exact sum without its zero terms (Z/ell^k is a ring; reduced representatives agree).
MapRes InducedMapEmbed(const Pres& X, const Pres& Y, uint32_t out_off, uint32_t in_off, uint32_t w, uint32_t rankF, uint32_t yc,
                       uint32_t xc) {
    // (audit F5) the shape assertion of the replaced code (f was sized yc x xc and InducedMap asserted f.r == Y.c, f.c == X.c)
    ENGINE_CHECK_MSG(Y.c == yc && X.c == xc, "InducedMap: the cochain map has the wrong shape");
    ENGINE_CHECK_MSG(uint64_t(out_off) + w <= Y.c && uint64_t(in_off) + w <= X.c, "InducedMapEmbed: the embedding leaves the cochains");
    auto dense_f = [&]() {
        Mat f(yc, xc);
        for (uint32_t t = 0; t < w; ++t) f.at(out_off + t, in_off + t) = 1;
        return f;
    };
    if (!par::Enabled(par::Lever::Embed)) return InducedMap(X, Y, dense_f(), rankF);
    // the validation's caller capture keeps seeing every call it would have stored (it skips f above max_entries anyway)
    if (uint64_t(Y.c) * X.c <= (uint64_t(1) << 20)) CallerCaptureInducedMap(X, Y, dense_f(), rankF);
    par::NoteEmbedMap();
    MapRes R;
    R.rankF = rankF;
    ENGINE_CHECK_MSG(X.ok && Y.ok, "InducedMap: a presentation was refused");
    const ZMod Z = MinZ(X.Z, Y.Z);
    R.Z = Z;
    const uint32_t gX = X.g(), gY = Y.g();
    Mat Gw(w, gX), Cw(gY, w);   // (gensT[:, in_off ..))^T reduced; coord[:, out_off ..) reduced
    for (uint32_t j = 0; j < gX; ++j) {
        const uint32_t* g = X.gensT.row(j) + in_off;
        for (uint32_t t = 0; t < w; ++t) Gw.at(t, j) = Z.red(g[t]);
    }
    for (uint32_t i = 0; i < gY; ++i) {
        const uint32_t* c = Y.coord.row(i) + out_off;
        uint32_t* o = Cw.row(i);
        for (uint32_t t = 0; t < w; ++t) o[t] = Z.red(c[t]);
    }
    R.F = DescentSmithBlockedEnabled() ? MulMod(Z, Cw, Gw) : Mul(Z, Cw, Gw);
    return InducedMapTail(X, Y, std::move(R));
}

namespace {
MapRes InducedMapTail(const Pres& X, const Pres& Y, MapRes R) {
    const ZMod Z = R.Z;
    const uint32_t rankF = R.rankF;
    const uint32_t gX = X.g(), gY = Y.g(), tX = static_cast<uint32_t>(X.tors.size()), tY = static_cast<uint32_t>(Y.tors.size());
    const bool blk = DescentSmithBlockedEnabled();
    for (uint32_t i = 0; i < tY; ++i) {
        const ZMod Zi(Z.ell, Y.tors[i]);
        for (uint32_t j = 0; j < gX; ++j) R.F.at(i, j) = Zi.red(R.F.at(i, j));
    }
    // the relations of X go to relations of Y (f is a cochain map; a failure is a defect, never a rounding)
    for (uint32_t j = 0; j < tX; ++j)
        for (uint32_t i = 0; i < gY; ++i) {
            const uint64_t x = R.F.at(i, j);
            if (!x) continue;
            // a free row of Y: the exact value is 0 and F is exact mod ell^k, so x itself must vanish (D2 audit F4; the
            // weaker x ell^{a_j} = 0 would let a defect x = ell^{k - a_j} u through)
            const bool ok = i < tY ? Z.mul(static_cast<uint32_t>(x), Z.pw(X.tors[j])) % Z.pw(Y.tors[i]) == 0 : false;
            if (!ok) {
                R.ok = false;
                R.why = "the cochain map does not carry the relations of the source into those of the target";
                return R;
            }
        }
    // coker f = coker [F | Delta_Y]
    Mat Mc(gY, gX + tY);
    for (uint32_t i = 0; i < gY; ++i)
        for (uint32_t j = 0; j < gX; ++j) Mc.at(i, j) = R.F.at(i, j);
    for (uint32_t i = 0; i < tY; ++i) Mc.at(i, gX + i) = Z.pw(Y.tors[i]);
    const uint32_t rM = tY + rankF;
    if (rankF > std::min(X.free, Y.free)) {
        R.ok = false;
        R.why = "rank input above the free ranks";
        return R;
    }
    // PAR-D2: on the lever, the factored form and only the rows >= rM of R^T (the kernel basis); else the oracle's dense R^T
    Smith S3;
    SmithFactors F3;
    if (blk)
        F3 = FactorCounted(Z, Mc, true);
    else
        S3 = SmithForm(Z, Mc, false, true);
    const std::vector<uint32_t>& sig3 = blk ? F3.sigma : S3.sigma;
    const uint32_t rank3 = static_cast<uint32_t>(sig3.size());
    if (rank3 != rM) {
        R.ok = false;
        R.why = "[F | Delta_Y] not certified (r_K = " + std::to_string(rank3) + ", rank_Q = " + std::to_string(rM) + ")";
        R.precision = true;
        return R;
    }
    R.coker.free_rank = gY - rM;
    AddTorsion(R.coker, Z.ell, sig3);
    const uint32_t sm = blk ? F3.smax() : S3.smax();
    if (Z.k <= sm + 1) {
        R.ok = false;
        R.why = "precision exhausted by [F | Delta_Y]";
        R.precision = true;
        return R;
    }
    const ZMod ZK(Z.ell, Z.k - sm);
    const uint32_t kL = gX + tY - rM;
    R.Lf = Mat(gX, kL);
    if (blk) {
        std::vector<uint32_t> krows(kL);
        for (uint32_t t = 0; t < kL; ++t) krows[t] = rM + t;
        const Mat RTk = FactorsRTRows(F3, krows);
        for (uint32_t t = 0; t < kL; ++t)
            for (uint32_t i = 0; i < gX; ++i) R.Lf.at(i, t) = ZK.red(RTk.at(t, i));
    } else {
        for (uint32_t t = 0; t < kL; ++t)
            for (uint32_t i = 0; i < gX; ++i) R.Lf.at(i, t) = ZK.red(S3.RT.at(rM + t, i));
    }
    R.rankL = gX - rankF;
    R.ZL = ZK;
    LatQuot Q = LatticeQuotient(ZK, R.Lf, R.rankL, DeltaOf(X, ZK), tX);
    if (!Q.ok) {
        R.ok = false;
        R.why = "kernel: " + Q.why;
        R.precision = Q.precision;
        return R;
    }
    R.ker = Q.LmodI;
    R.im = Q.ZmodL;
    return R;
}

}  // namespace

bool ExactAt(const Pres& X, const MapRes& g, const MapRes& h, std::string& why) {
    if (!g.ok || !h.ok) {
        why = "a map was refused";
        return false;
    }
    const ZMod Z = MinZ(g.Z, h.ZL);
    const uint32_t gX = X.g(), tX = static_cast<uint32_t>(X.tors.size());
    ENGINE_CHECK(g.F.r == gX && h.Lf.r == gX);
    Mat I(gX, g.F.c + tX);
    for (uint32_t i = 0; i < gX; ++i)
        for (uint32_t j = 0; j < g.F.c; ++j) I.at(i, j) = Z.red(g.F.at(i, j));
    for (uint32_t i = 0; i < tX; ++i) I.at(i, g.F.c + i) = Z.pw(X.tors[i]);
    LatQuot Q = LatticeQuotient(Z, Reduce(h.Lf, Z), h.rankL, I, tX + g.rankF);
    if (!Q.ok) {
        why = Q.why;
        return false;
    }
    if (!Q.LmodI.is_zero()) {
        why = "ker h / im g = " + Q.LmodI.to_string();
        return false;
    }
    return true;
}

bool IdentityOnCohomology(const Pres& Y, const Mat& F, std::string& why) {
    ENGINE_CHECK(F.r == Y.g());
    const uint32_t tY = static_cast<uint32_t>(Y.tors.size());
    for (uint32_t i = 0; i < F.r; ++i)
        for (uint32_t j = 0; j < F.c; ++j) {
            const uint32_t x = F.at(i, j);
            const bool zero = i < tY ? x % Y.Z.pw(Y.tors[i]) == 0 : x == 0;
            if (!zero) {
                why = "entry (" + std::to_string(i) + "," + std::to_string(j) + ") not in the relations";
                return false;
            }
        }
    return true;
}

// ---------------------------------------------------------------------------
// the pair: local pieces
// ---------------------------------------------------------------------------
namespace {

// F_a F_b - s I (s >= 0) or F_a F_b - F_c (c given) in Y's coordinates, at the smaller precision.
Mat Combine(const ZMod& Z0, const Mat& Fa, const Mat& Fb, uint32_t s, const std::vector<const Mat*>& minus) {
    Mat P = Mul(Z0, Reduce(Fa, Z0), Reduce(Fb, Z0));
    if (s) {
        ENGINE_CHECK(P.r == P.c);
        for (uint32_t i = 0; i < P.r; ++i) P.at(i, i) = Z0.sub(P.at(i, i), Z0.red(s));
    }
    for (const Mat* M : minus) {
        ENGINE_CHECK(M->r == P.r && M->c == P.c);
        for (size_t x = 0; x < P.a.size(); ++x) P.a[x] = Z0.sub(P.a[x], Z0.red(M->a[x]));
    }
    return P;
}

struct Local {
    uint32_t ell = 0, qt = 0;
    std::vector<Pres> HA, HB, HAB;          // q = 0 .. qt
    std::vector<MapRes> res, cor, j, dl;    // res, cor, dl: 0 .. qt; j: 0 .. qt - 1
};

struct Betti {
    uint32_t A[2] = {0, 0}, B[2] = {0, 0};
    uint32_t a(uint32_t q) const { return q < 2 ? A[q] : 0u; }
    uint32_t b(uint32_t q) const { return q < 2 ? B[q] : 0u; }
    uint32_t beta(int q) const { return (q == 0 || q == 1) ? B[q] - A[q] : 0u; }
};

// The reduced complex of Iso^(ell) for one module, in J-coordinates on both sides.
struct RedComplex {
    uint32_t D = 0, q_top = 0;
    std::vector<std::vector<std::vector<uint32_t>>> J, R;   // [q][cell]
    std::vector<std::vector<const integral_model::ColumnSpec*>> col;
    std::vector<std::vector<uint32_t>> off;
    std::vector<std::map<std::pair<uint32_t, uint32_t>, Mat>> Ginv;   // [q] (s, rep) -> G^{-1} (only q_order > 1)
    std::vector<std::vector<std::pair<uint32_t, uint32_t>>> key;
    std::vector<uint32_t> c;
    std::vector<Mat> d;                                       // d[q]: c[q+1] x c[q]
};

RedComplex BuildRed(const LocalModel& L, const U32Ctx& ctx, const ZMod& Z, const CoefficientModule<uint32_t>& M, uint32_t q_top) {
    RedComplex X;
    X.D = M.dim;
    X.q_top = q_top;
    std::map<integral_model::GKey, DenseBuf<uint32_t>> memo;
    std::map<std::pair<uint32_t, uint32_t>, std::pair<std::vector<uint32_t>, std::vector<uint32_t>>> sel;
    std::map<std::pair<uint32_t, uint32_t>, Mat> ginv;
    X.J.resize(q_top + 1);
    X.R.resize(q_top + 1);
    X.col.resize(q_top + 1);
    X.off.resize(q_top + 1);
    X.key.resize(q_top + 1);
    X.Ginv.resize(1);
    for (uint32_t q = 0; q <= q_top; ++q) {
        uint32_t c = 0;
        for (const auto& cell : L.cells(q)) {
            const auto k = std::make_pair(cell.s, cell.rep);
            const auto& cs = L.column(cell.s, cell.rep);
            auto it = sel.find(k);
            if (it == sel.end()) {
                std::vector<uint32_t> J, R;
                SelectBlockJR(ctx, cs, M, memo, J, R);
                if (cs.q_order > 1) {
                    const DenseBuf<uint32_t> P = integral_model::RhoQ(ctx, cs.e, M, memo);
                    Mat G(static_cast<uint32_t>(R.size()), static_cast<uint32_t>(J.size()));
                    for (uint32_t a = 0; a < R.size(); ++a)
                        for (uint32_t b = 0; b < J.size(); ++b) G.at(a, b) = P.at(R[a], J[b]);
                    ginv.emplace(k, InverseUnit(Z, G));
                }
                it = sel.emplace(k, std::make_pair(J, R)).first;
            }
            X.J[q].push_back(it->second.first);
            X.R[q].push_back(it->second.second);
            X.col[q].push_back(&cs);
            X.off[q].push_back(c);
            X.key[q].push_back(k);
            c += static_cast<uint32_t>(it->second.first.size());
        }
        X.c.push_back(c);
    }
    X.Ginv[0] = std::move(ginv);
    for (uint32_t q = 0; q < q_top; ++q) {
        Mat A = FromDense(ClauseMatrixReduced(L, ctx, M, q, X.J[q], X.R[q + 1], X.col[q], X.col[q + 1]));
        ENGINE_CHECK(A.r == X.c[q + 1] && A.c == X.c[q]);
        // pi_R coordinates -> J coordinates on the target: G_y^{-1} per target cell with Q != 1
        for (uint32_t y = 0; y < X.col[q + 1].size(); ++y) {
            if (X.col[q + 1][y]->q_order == 1) continue;
            const Mat& Gi = X.Ginv[0].at(X.key[q + 1][y]);
            const uint32_t o = X.off[q + 1][y], w = Gi.r;
            Mat blk(w, A.c);
            for (uint32_t a = 0; a < w; ++a) std::copy(A.row(o + a), A.row(o + a) + A.c, blk.row(a));
            const Mat nb = Mul(Z, Gi, blk);
            for (uint32_t a = 0; a < w; ++a) std::copy(nb.row(a), nb.row(a) + A.c, A.row(o + a));
        }
        X.d.push_back(std::move(A));
    }
    return X;
}

// A fibre map f : M_X -> M_Y (D_Y x D_X) on the reduced cochains of degree q, block by block ([C, prop:descentmaps](1)(b)).
Mat ReduceFibre(const RedComplex& X, const RedComplex& Y, const CoefficientModule<uint32_t>& MX, const U32Ctx& ctx,
                const ZMod& Z, const Mat& f, uint32_t q, std::map<std::pair<uint32_t, uint32_t>, Mat>& memo_blocks) {
    ENGINE_CHECK(f.r == Y.D && f.c == X.D && X.col[q].size() == Y.col[q].size());
    Mat out(Y.c[q], X.c[q]);
    std::map<integral_model::GKey, DenseBuf<uint32_t>> memo;
    for (uint32_t x = 0; x < X.col[q].size(); ++x) {
        const auto k = X.key[q][x];
        auto it = memo_blocks.find(k);
        if (it == memo_blocks.end()) {
            const auto& J = X.J[q][x];
            const auto& R = Y.R[q][x];
            Mat blk;
            if (X.col[q][x]->q_order == 1) {
                blk = f;
            } else {
                const DenseBuf<uint32_t> P = integral_model::RhoQ(ctx, X.col[q][x]->e, MX, memo);
                Mat PJ(X.D, static_cast<uint32_t>(J.size()));
                for (uint32_t i = 0; i < X.D; ++i)
                    for (uint32_t b = 0; b < J.size(); ++b) PJ.at(i, b) = P.at(i, J[b]);
                const Mat W = Mul(Z, f, PJ);   // D_Y x |J|
                Mat WR(static_cast<uint32_t>(R.size()), W.c);
                for (uint32_t a = 0; a < R.size(); ++a) std::copy(W.row(R[a]), W.row(R[a]) + W.c, WR.row(a));
                blk = Mul(Z, Y.Ginv[0].at(Y.key[q][x]), WR);
            }
            it = memo_blocks.emplace(k, std::move(blk)).first;
        }
        const Mat& B = it->second;
        const uint32_t ro = Y.off[q][x], co = X.off[q][x];
        ENGINE_CHECK(B.r == Y.J[q][x].size() && B.c == X.J[q][x].size());
        for (uint32_t a = 0; a < B.r; ++a) std::copy(B.row(a), B.row(a) + B.c, out.row(ro + a) + co);
    }
    return out;
}

// (D5) THE TREE ROUTE ([C, prop:isomodeltree](5): Iso^(ell)_T satisfies (T1)-(T3) of prop:descentmaps(5), so (1), (2), (4) there
// hold on it verbatim; a fibre map acts block by block on each M^{Q_x}).  The reduced complex of Iso^(ell)_T (TreeCochains.h,
// (2)(a), (b)) in J-coordinates on both sides: the closed-form d^q has pi_R coordinates on the target, turned into J-coordinates
// by G_x^{-1}, G_x = rho(e_x)[R_x, J_x], exactly as BuildRed does on Iso^(ell).  Columns per degree: ell = 2: S, U (q = 0),
// S, U, E (q >= 1); ell = 3: S, U (q = 0), U, E (q = 1), U (q >= 2).  Every block is formed from the block-monomial action
// (no dense D x D rho is built).
struct TreeRed {
    RedComplex X;                                         // c, d (J/R/col of RedComplex unused on this route)
    uint32_t ell = 0;
    std::vector<std::vector<char>> cols;                  // [q] column letters
    std::map<char, integral_model::ImageSelection> sel;   // M^{Q_x}
    std::map<char, integral_model::QGroupRingElt> e;      // e_x
    std::map<char, Mat> Ginv;                             // G_x^{-1} for the columns with Q_x acting non-trivially
};
TreeRed BuildRedTree(uint32_t ell, const U32Ctx& ctx, const ZMod& Z, const integral_model::TreeModule<uint32_t>& M, uint32_t q_top) {
    namespace te = integral_model::tree_elt;
    TreeRed T;
    T.ell = ell;
    T.X.D = M.dim();
    T.X.q_top = q_top;
    const integral_model::TreeLocalForm<uint32_t> F = integral_model::MakeTreeLocalForm(ell, ctx, M, q_top);
    if (ell == 2) {
        T.sel = {{'S', F.sel_S}, {'U', F.sel_U}, {'E', F.sel_E}};
        T.e = {{'S', te::one()}, {'U', te::eU2()}, {'E', te::one()}};
    } else {
        ENGINE_CHECK(ell == 3);
        T.sel = {{'S', F.sel_S}, {'U', F.sel_U}, {'E', F.sel_E}};
        T.e = {{'S', te::eS4()}, {'U', te::ec()}, {'E', te::ec()}};
    }
    for (uint32_t q = 0; q <= q_top; ++q) {
        std::vector<char> c;
        if (ell == 2) c = q == 0 ? std::vector<char>{'S', 'U'} : std::vector<char>{'S', 'U', 'E'};
        else c = q == 0 ? std::vector<char>{'S', 'U'} : (q == 1 ? std::vector<char>{'U', 'E'} : std::vector<char>{'U'});
        uint32_t w = 0;
        for (char x : c) w += T.sel.at(x).rank();
        ENGINE_CHECK_MSG(w == F.cochain_rank(q), "BuildRedTree: the column layout disagrees with prop:isomodeltree(2)(e)");
        T.cols.push_back(c);
        T.X.c.push_back(w);
    }
    for (const auto& [x, sl] : T.sel) {
        if (sl.all) continue;
        const uint32_t r = sl.rank();
        std::vector<uint32_t> g(size_t(r) * r, 0);
        integral_model::AddRhoBlock(ctx, g, r, 0, 0, M, T.e.at(x), sl, sl);   // G = rho(e)[R, J]
        T.Ginv.emplace(x, InverseUnit(Z, FromRowMajor(r, r, g)));
    }
    for (uint32_t q = 0; q < q_top; ++q) {
        const uint32_t rep = F.representative(q);
        if (rep < q) {   // (2)(d): d^q == d^rep as matrices
            Mat copy = T.X.d[rep];
            T.X.d.push_back(std::move(copy));
            continue;
        }
        Mat A = FromRowMajor(T.X.c[q + 1], T.X.c[q], F.differential(ctx, M, q));
        uint32_t o = 0;
        for (char y : T.cols[q + 1]) {
            const uint32_t w = T.sel.at(y).rank();
            if (!T.sel.at(y).all && w && integral_model::ActiveTreeFault() != integral_model::TreeFault::D2DiffNoGinv) {
                Mat blk(w, A.c);
                for (uint32_t a = 0; a < w; ++a) std::copy(A.row(o + a), A.row(o + a) + A.c, blk.row(a));
                const Mat nb = Mul(Z, T.Ginv.at(y), blk);
                for (uint32_t a = 0; a < w; ++a) std::copy(nb.row(a), nb.row(a) + A.c, A.row(o + a));
            }
            o += w;
        }
        T.X.d.push_back(std::move(A));
    }
    return T;
}
// A fibre map f : M_X -> M_Y (D_Y x D_X) on the reduced tree cochains of degree q, block by block: G_Y^{-1} pi_R (f B_X).
Mat ReduceFibreTree(const TreeRed& X, const TreeRed& Y, const integral_model::TreeModule<uint32_t>& MX, const U32Ctx& ctx, const ZMod& Z,
                    const Mat& f, uint32_t q, std::map<char, Mat>& memo_blocks) {
    ENGINE_CHECK(f.r == Y.X.D && f.c == X.X.D && X.cols[q] == Y.cols[q]);
    Mat out(Y.X.c[q], X.X.c[q]);
    uint32_t ro = 0, co = 0;
    for (char x : X.cols[q]) {
        const auto& sx = X.sel.at(x);
        const auto& sy = Y.sel.at(x);
        auto it = memo_blocks.find(x);
        if (it == memo_blocks.end()) {
            Mat blk;
            if (sx.all && sy.all) {
                blk = f;
            } else {
                integral_model::ImageSelection all;
                all.all = true;
                all.rpos.resize(X.X.D);
                std::iota(all.rpos.begin(), all.rpos.end(), 0);
                std::vector<uint32_t> b(size_t(X.X.D) * sx.rank(), 0);
                integral_model::AddRhoBlock(ctx, b, sx.rank(), 0, 0, MX, X.e.at(x), all, sx);   // B_X = rho(e)[:, J]
                const Mat W = Mul(Z, f, FromRowMajor(X.X.D, sx.rank(), b));                      // D_Y x |J_X|
                Mat WR(sy.rank(), W.c);
                for (uint32_t a = 0; a < sy.rank(); ++a) std::copy(W.row(sy.R[a]), W.row(sy.R[a]) + W.c, WR.row(a));
                blk = (sy.all || integral_model::ActiveTreeFault() == integral_model::TreeFault::D2FibreNoGinv) ? WR : Mul(Z, Y.Ginv.at(x), WR);
            }
            it = memo_blocks.emplace(x, std::move(blk)).first;
        }
        const Mat& B = it->second;
        ENGINE_CHECK(B.r == sy.rank() && B.c == sx.rank());
        for (uint32_t a = 0; a < B.r; ++a) std::copy(B.row(a), B.row(a) + B.c, out.row(ro + a) + co);
        ro += sy.rank();
        co += sx.rank();
    }
    return out;
}

Mat Neg(const ZMod& Z, Mat A) {
    for (auto& x : A.a) x = Z.sub(0, Z.red(x));
    return A;
}
void Place(Mat& T, const Mat& B, uint32_t r0, uint32_t c0) {
    for (uint32_t i = 0; i < B.r; ++i) std::copy(B.row(i), B.row(i) + B.c, T.row(r0 + i) + c0);
}

struct PairCtx {
    uint32_t N = 1, MN = 1, n = 0, index = 0;
    bool normal = false;
    std::vector<hecke::Mat22> sigmas;     // coset representatives of B in A other than I (normal pairs)
    uint32_t G_order = 1, G_ab_order = 1;
    Betti bt;
    const PairOptions* opt = nullptr;
};

// Fibre matrices over ctx.
struct Fibres {
    Mat alpha, omega;
    std::vector<Mat> Rs;   // R_sigma on M_B (normal pairs)
};
Fibres BuildFibres(const PairCtx& pc, const U32Ctx& ctx) {
    Fibres F;
    const hecke::level::Gamma0LevelN A(pc.N), B(pc.MN);
    const auto chi = les::CharacterDatum::Trivial(ctx.p);
    les::RestrictionFibreMap rf(A, chi, B, chi, pc.n, ctx);
    ENGINE_CHECK_MSG(rf.index() == pc.index, "AssemblePairMaps: [A : B] from the fibre map differs from psi(MN) / psi(N)");
    // Gamma-equivariance of alpha and omega on the generators S, T (prop:descentmaps(1): u is a Z[Gamma]-map)
    for (const hecke::Mat22& g : {hecke::Mat22::S(), hecke::Mat22::T()}) {
        const auto MA = les::TwistedMonomialMatrix(A, chi, g, pc.n, ctx);
        const auto MB = les::TwistedMonomialMatrix(B, chi, g, pc.n, ctx);
        ENGINE_CHECK_MSG((rf.alpha() * MA).differing_entries(MB * rf.alpha()) == 0, "AssemblePairMaps: alpha is not a Gamma-map");
        ENGINE_CHECK_MSG((rf.trace() * MB).differing_entries(MA * rf.trace()) == 0, "AssemblePairMaps: omega is not a Gamma-map");
    }
    const uint32_t d = pc.n + 1;
    F.alpha = FromRowMajor(B.m() * d, A.m() * d, rf.alpha().dense());
    F.omega = FromRowMajor(A.m() * d, B.m() * d, rf.trace().dense());
    if (pc.normal) {
        const auto pB = les::PermutationOf(B);
        for (const auto& s : pc.sigmas) {
            les::ConjugationFibreMap R(A, chi, B, pB, s, pc.n, ctx);
            F.Rs.push_back(FromRowMajor(B.m() * d, B.m() * d, R.matrix().dense()));
        }
    }
    if (pc.opt->test_break_alpha) F.alpha.at(0, 0) = static_cast<uint32_t>((uint64_t(F.alpha.at(0, 0)) + 1) % ctx.p);
    return F;
}

// direct sum of k copies of a presentation (for the invariants of a normal pair)
Pres DirectSum(const Pres& X, uint32_t k) {
    Pres S = ZeroPres(X.Z, X.c * k);
    const uint32_t t = static_cast<uint32_t>(X.tors.size()), g = X.g();
    S.gensT = Mat(g * k, X.c * k);
    S.coord = Mat(g * k, X.c * k);
    // order: all torsion generators first (copy by copy), then all free ones
    for (uint32_t i = 0; i < k; ++i)
        for (uint32_t a = 0; a < t; ++a) S.tors.push_back(X.tors[a]);
    S.free = X.free * k;
    for (uint32_t i = 0; i < k; ++i)
        for (uint32_t a = 0; a < g; ++a) {
            const uint32_t row = a < t ? i * t + a : k * t + i * X.free + (a - t);
            std::copy(X.gensT.row(a), X.gensT.row(a) + X.c, S.gensT.row(row) + i * X.c);
            std::copy(X.coord.row(a), X.coord.row(a) + X.c, S.coord.row(row) + i * X.c);
        }
    return S;
}

// The checks shared by both routes: exactness, identities, the normal case.  qt = the piece's top degree.
bool CheckPiece(const PairCtx& pc, Local& L, const std::vector<std::vector<Mat>>& Rs_red /* [sigma][q] cochain maps on B */,
                PieceReport& rep, std::string& five) {
    std::string why;
    auto fail = [&](const std::string& w) {
        rep.ok = false;
        rep.why = w;
        return false;
    };
    for (uint32_t q = 0; q <= L.qt; ++q) {
        if (!L.HA[q].ok) return fail("H^" + std::to_string(q) + "(A): " + L.HA[q].why);
        if (!L.HB[q].ok) return fail("H^" + std::to_string(q) + "(B): " + L.HB[q].why);
        if (!L.HAB[q].ok) return fail("H^" + std::to_string(q) + "(A,B): " + L.HAB[q].why);
    }
    for (uint32_t q = 0; q <= L.qt; ++q) {
        if (!L.res[q].ok) return fail("res^" + std::to_string(q) + ": " + L.res[q].why);
        if (!L.cor[q].ok) return fail("cor^" + std::to_string(q) + ": " + L.cor[q].why);
        if (!L.dl[q].ok) return fail("dl^" + std::to_string(q) + ": " + L.dl[q].why);
        if (q < L.qt && !L.j[q].ok) return fail("j^" + std::to_string(q) + ": " + L.j[q].why);
    }
    // exactness: H^0(A), then H^q(B), H^{q+1}(A,B), H^{q+1}(A)
    if (!ExactAt(L.HA[0], L.dl[0], L.res[0], why)) return fail("not exact at H^0(A): " + why);
    ++rep.exact_spots;
    for (uint32_t q = 0; q < L.qt; ++q) {
        if (!ExactAt(L.HB[q], L.res[q], L.j[q], why)) return fail("not exact at H^" + std::to_string(q) + "(B): " + why);
        if (!ExactAt(L.HAB[q + 1], L.j[q], L.dl[q + 1], why))
            return fail("not exact at H^" + std::to_string(q + 1) + "(A,B): " + why);
        if (!ExactAt(L.HA[q + 1], L.dl[q + 1], L.res[q + 1], why))
            return fail("not exact at H^" + std::to_string(q + 1) + "(A): " + why);
        rep.exact_spots += 3;
    }
    // identities on cohomology: cor o res = [A : B] (prop:descentmaps(1)(d))
    for (uint32_t q = 0; q <= L.qt; ++q) {
        const ZMod Z0 = MinZ(L.res[q].Z, L.cor[q].Z);
        if (!IdentityOnCohomology(L.HA[q], Combine(Z0, L.cor[q].F, L.res[q].F, pc.index, {}), why))
            return fail("cor o res != [A:B] on H^" + std::to_string(q) + ": " + why);
        ++rep.identity_checks;
        // the split case: ell !| [A : B] => res injective, cor onto, H(B) = res H(A) + ker cor
        if (pc.index % L.ell != 0) {
            AbelianGroupType sum = L.HA[q].type();
            const AbelianGroupType& kc = L.cor[q].ker;
            sum.free_rank += kc.free_rank;
            for (const auto& [p, ex] : kc.torsion)
                for (uint32_t a : ex) sum.add_torsion(p, a);
            if (!L.res[q].ker.is_zero() || !L.cor[q].coker.is_zero() || sum != L.HB[q].type())
                return fail("split case (ell !| [A:B]) fails on H^" + std::to_string(q));
            ++rep.identity_checks;
        }
    }
    // the normal case (prop:descentmaps(4)(a), (b))
    if (pc.normal) {
        for (uint32_t q = 0; q <= L.qt; ++q) {
            std::vector<MapRes> cs;
            for (size_t s = 0; s < Rs_red.size(); ++s) {
                cs.push_back(InducedMap(L.HB[q], L.HB[q], Rs_red[s][q], L.HB[q].free));
                if (!cs.back().ok) return fail("c_sigma on H^" + std::to_string(q) + "(B): " + cs.back().why);
                const ZMod Z0 = MinZ(cs.back().Z, L.res[q].Z);
                if (!IdentityOnCohomology(L.HB[q], Combine(Z0, cs.back().F, L.res[q].F, 0, {&L.res[q].F}), why))
                    return fail("c_sigma o res != res on H^" + std::to_string(q) + ": " + why);
                ++rep.identity_checks;
            }
            // res o cor = sum_{sigma in G} c_sigma (the identity included)
            ZMod Z0 = MinZ(L.res[q].Z, L.cor[q].Z);
            for (const auto& c : cs) Z0 = MinZ(Z0, c.Z);
            std::vector<const Mat*> minus;
            for (const auto& c : cs) minus.push_back(&c.F);
            Mat P = Combine(Z0, L.res[q].F, L.cor[q].F, 1, minus);   // res cor - I - sum_{sigma != 1} c_sigma
            if (!IdentityOnCohomology(L.HB[q], P, why)) return fail("res o cor != Nm_G on H^" + std::to_string(q) + ": " + why);
            ++rep.identity_checks;
        }
        // (4)(b) in degree 1: 0 -> H^1(G; H^0(B)) -> H^1(A) -> H^1(B)^G -> H^2(G; H^0(B)); H^1(G; H^0(B)) = 0 (H^0(B) = 0 for
        // n > 0, = Z with G finite for n = 0)
        if (L.qt >= 1 && !Rs_red.empty()) {
            const uint32_t k = static_cast<uint32_t>(Rs_red.size());
            const Pres& HB1 = L.HB[1];
            const Pres S = DirectSum(HB1, k);
            Mat f(S.c, HB1.c);
            for (uint32_t s = 0; s < k; ++s) {
                const Mat& R = Rs_red[s][1];
                for (uint32_t i = 0; i < HB1.c; ++i)
                    for (uint32_t jj = 0; jj < HB1.c; ++jj) {
                        uint32_t x = HB1.Z.red(R.at(i, jj));
                        if (i == jj) x = HB1.Z.sub(x, 1);
                        f.at(s * HB1.c + i, jj) = x;
                    }
            }
            const MapRes inv = InducedMap(HB1, S, f, uint32_t(int64_t(pc.bt.b(1) - pc.bt.a(1)) + pc.opt->test_inv_rank_offset));
            if (!inv.ok) return fail("the invariants map on H^1(B): " + inv.why);
            const ZMod Z0 = MinZ(inv.ZL, L.res[1].Z);
            const uint32_t tB = static_cast<uint32_t>(HB1.tors.size());
            Mat I(HB1.g(), L.res[1].F.c + tB);
            for (uint32_t i = 0; i < HB1.g(); ++i)
                for (uint32_t jj = 0; jj < L.res[1].F.c; ++jj) I.at(i, jj) = Z0.red(L.res[1].F.at(i, jj));
            for (uint32_t i = 0; i < tB; ++i) I.at(i, L.res[1].F.c + i) = Z0.pw(HB1.tors[i]);
            LatQuot Q = LatticeQuotient(Z0, Reduce(inv.Lf, Z0), inv.rankL, I, tB + pc.bt.a(1));
            if (!Q.ok) return fail("H^1(B)^G / res H^1(A): " + Q.why);
            if (!L.res[1].ker.is_zero()) return fail("five-term: res^1 not injective");
            const AbelianGroupType& C = Q.LmodI;
            std::ostringstream o;
            o << "ell=" << L.ell << ": ker res^1 = 0, H^1(B)^G / res H^1(A) = " << C.to_string();
            if (pc.n > 0) {
                if (!C.is_zero()) return fail("five-term (n > 0): res^1 is not onto H^1(B)^G: " + C.to_string());
            } else {
                uint64_t order = 1;
                bool killed = C.free_rank == 0;
                for (const auto& [p, ex] : C.torsion)
                    for (uint32_t a : ex) {
                        uint64_t pa = 1;
                        for (uint32_t i = 0; i < a; ++i) pa *= p;
                        order *= pa;
                        if (pc.G_order % pa != 0) killed = false;
                    }
                if (!killed || pc.G_ab_order % order != 0)
                    return fail("five-term (n = 0): H^1(B)^G / res H^1(A) = " + C.to_string() + " does not embed in H^2(G; Z)");
                o << " (divides |G^ab| = " << pc.G_ab_order << ")";
            }
            ++rep.identity_checks;
            five += (five.empty() ? "" : "; ") + o.str();
        }
    }
    return true;
}

uint64_t RowsOf(uint32_t ell, uint32_t q, uint32_t D) {
    // the reduced cochain ranks of Iso^(ell) per unit of D (DescentAssembly's Sec. 3.1 figures), rounded up
    static const double w2[] = {4.34, 13.34, 18.34, 18.34, 18.34, 18.34, 18.34};
    static const double w3[] = {4.5, 10.0, 6.0, 1.0, 1.0, 1.0, 1.0};
    return static_cast<uint64_t>((ell == 2 ? w2[q] : w3[q]) * D) + 1;
}

}  // namespace

uint64_t PairRssStructural(uint32_t ell, uint32_t DA, uint32_t DB, uint32_t q_top) {
    ENGINE_CHECK(ell == 2 || ell == 3);
    // stored: d_A, d_B and the cone's D^j (uint32_t), the fibre maps per degree; working: the largest Smith form with left
    // transforms (A + L + L^{-1}: three c x c); the dense alpha / omega / R_sigma (D_B x D_A) and the rho memo.
    uint64_t stored = 0, cmax = 0;
    for (uint32_t q = 0; q < q_top; ++q) {
        stored += RowsOf(ell, q + 1, DA) * RowsOf(ell, q, DA) + RowsOf(ell, q + 1, DB) * RowsOf(ell, q, DB);
        stored += 2 * RowsOf(ell, q, DB) * RowsOf(ell, q, DA);   // alpha, omega reduced
    }
    for (int j = -1; j + 1 < static_cast<int>(q_top); ++j) {
        const uint64_t cj = RowsOf(ell, j + 1, DA) + (j >= 0 ? RowsOf(ell, j, DB) : 0);
        const uint64_t cj1 = RowsOf(ell, j + 2, DA) + RowsOf(ell, j + 1, DB);
        stored += cj * cj1;
        cmax = std::max(cmax, std::max(cj, cj1));
    }
    const uint64_t work = 3 * cmax * cmax;
    return 4ull * (stored + work) + 4ull * 4 * uint64_t(DA) * DB + 48ull * 4 * uint64_t(DB) * DB;
}

uint64_t PairRssStructuralTree(uint32_t ell, uint32_t DA, uint32_t DB, uint32_t q_top) {
    ENGINE_CHECK(ell == 2 || ell == 3);
    // (D5) the same count on the tree's reduced ranks ([C, prop:isomodeltree](2)(e): 4/3, 7/3, 7/3, ... D at 2, r_{U^2} <= D/3;
    // 1.5, 2, 1, 1, ... D at 3); no dense rho memo on this route (block-monomial); the dense alpha / omega / R_sigma stay.
    auto rows = [ell](uint32_t q, uint32_t D) -> uint64_t {
        if (ell == 2) return (q == 0 ? 4ull * D / 3 : 7ull * D / 3) + 2;
        return (q == 0 ? 3ull * D / 2 : (q == 1 ? 2ull * D : uint64_t(D))) + 2;
    };
    uint64_t stored = 0, cmax = 0;
    for (uint32_t q = 0; q < q_top; ++q) {
        stored += rows(q + 1, DA) * rows(q, DA) + rows(q + 1, DB) * rows(q, DB);
        stored += 2 * rows(q, DB) * rows(q, DA);
    }
    for (int j = -1; j + 1 < static_cast<int>(q_top); ++j) {
        const uint64_t cj = rows(j + 1, DA) + (j >= 0 ? rows(j, DB) : 0);
        const uint64_t cj1 = rows(j + 2, DA) + rows(j + 1, DB);
        stored += cj * cj1;
        cmax = std::max(cmax, std::max(cj, cj1));
    }
    const uint64_t work = 3 * cmax * cmax;
    return 4ull * (stored + work) + 4ull * 4 * uint64_t(DA) * DB + 4ull * 4 * uint64_t(DB) * DB;
}

uint64_t PredictPairRssTree(uint32_t ell, uint32_t DA, uint32_t DB, uint32_t q_top) {
    // (D5) the calibrated factor on the tree's structural count (DESCENT5_report.md): the smallest multiple of 1/20 with >= 15 %
    // headroom on every measured tree cell.
    constexpr uint64_t kNum = 38;   // 38/20 = 19/10
    return PairRssStructuralTree(ell, DA, DB, q_top) / 20 * kNum + (256ull << 20);
}

uint64_t PredictPairRss(uint32_t ell, uint32_t DA, uint32_t DB, uint32_t q_top) {
    // The structural count under-predicts the measured growth of the piece (1.2x - 2.1x; probably the b > 0 degrees' two more c x c
    // transforms and, at some (N, n), reduced ranks above the per-D averages of RowsOf; not profiled).  Calibrated factors (D2 audit F3;
    // DESCENT2_predictions.md, amendment 2 and its results; DESCENT2_report.md "Audit fixes"): one per class, the smallest multiple of
    // 1/20 that leaves >= 15 % headroom (growth <= pred / 1.15) on EVERY measured cell of the class -- 23 calibration pieces (binary
    // of 8f698ae8) and the five reach cells (binary 18129bdf):
    //   ell = 2, [A:B] = D_B / D_A = 2: 17/10 (worst (30, 60, 6), S = 1512); ell = 2, [A:B] > 2: 19/10 (worst (21, 63, 2), (4, 16, 6));
    //   ell = 3: 41/20 with the 256 MiB allowance NOT credited (measured only to S = 392; worst (21, 63, 2)); the ell = 3 piece is never
    //   the binding one (its count is ~ 1/5 of ell = 2's).  The 8/5 of the first recalibration left 2 % at (4, 16, 6) and was exceeded
    //   at (21, 63, 2).
    const uint64_t quad = PairRssStructural(ell, DA, DB, q_top);
    const uint64_t num = ell == 3 ? 41 : (DB > 2ull * DA ? 38 : 34), den = 20;
    return quad / den * num + (256ull << 20);
}

// ---------------------------------------------------------------------------
// the assembly
// ---------------------------------------------------------------------------
namespace {

bool DensePiece(const PairCtx& pc, uint32_t ell, uint32_t qt, Local& L, PieceReport& rep, std::string& five) {
    const uint32_t K = LocalWidth(ell);
    const U32Ctx ctx = U32Ctx::prime_power(ell, K);
    const ZMod Z(ell, K);
    const LocalModel& LM = ell == 2 ? Iso2() : Iso3();
    ENGINE_CHECK(qt <= LM.q_top());
    const Fibres F = BuildFibres(pc, ctx);
    const Betti& bt = pc.bt;
    // reduced fibre maps per degree
    std::vector<Mat> al, om;
    std::vector<std::vector<Mat>> Rr(F.Rs.size());
    // (D5) the tree route ([C, prop:isomodeltree](5)) unless HECKE_DESCENT_TREE=0 (the dense Iso^(ell) rollback)
    const bool tree = DescentTreeEnabled();
    RedComplex XA, XB;
    if (tree) {
        const auto TMA = integral_model::TreeInducedSymModule(pc.N, pc.n, ctx), TMB = integral_model::TreeInducedSymModule(pc.MN, pc.n, ctx);
        TreeRed TA = BuildRedTree(ell, ctx, Z, TMA, qt), TB = BuildRedTree(ell, ctx, Z, TMB, qt);
        std::map<char, Mat> ma, mo;
        std::vector<std::map<char, Mat>> mr(F.Rs.size());
        for (uint32_t q = 0; q <= qt; ++q) {
            al.push_back(ReduceFibreTree(TA, TB, TMA, ctx, Z, F.alpha, q, ma));
            om.push_back(ReduceFibreTree(TB, TA, TMB, ctx, Z, F.omega, q, mo));
            for (size_t s = 0; s < F.Rs.size(); ++s) Rr[s].push_back(ReduceFibreTree(TB, TB, TMB, ctx, Z, F.Rs[s], q, mr[s]));
        }
        XA = std::move(TA.X);
        XB = std::move(TB.X);
    } else {
    const CoefficientModule<uint32_t> MA = InducedSymModule(pc.N, pc.n, ctx), MB = InducedSymModule(pc.MN, pc.n, ctx);
    XA = BuildRed(LM, ctx, Z, MA, qt);
    XB = BuildRed(LM, ctx, Z, MB, qt);
    {
        std::map<std::pair<uint32_t, uint32_t>, Mat> ma, mo;
        std::vector<std::map<std::pair<uint32_t, uint32_t>, Mat>> mr(F.Rs.size());
        for (uint32_t q = 0; q <= qt; ++q) {
            al.push_back(ReduceFibre(XA, XB, MA, ctx, Z, F.alpha, q, ma));
            om.push_back(ReduceFibre(XB, XA, MB, ctx, Z, F.omega, q, mo));
            for (size_t s = 0; s < F.Rs.size(); ++s) Rr[s].push_back(ReduceFibre(XB, XB, MB, ctx, Z, F.Rs[s], q, mr[s]));
        }
    }
    }
    // rational ranks of d_A, d_B (Betti recursion, [C, prop:certifiedeell](4)(c)(ii))
    auto ranks = [&](const RedComplex& X, bool isA) {
        std::vector<uint32_t> r;
        int64_t prev = 0;
        for (uint32_t q = 0; q < qt; ++q) {
            const int64_t x = int64_t(X.c[q]) - int64_t(isA ? bt.a(q) : bt.b(q)) - prev;
            ENGINE_CHECK_MSG(x >= 0, "AssemblePairMaps: negative Betti-recursion rank");
            r.push_back(static_cast<uint32_t>(x));
            prev = x;
        }
        return r;
    };
    const std::vector<uint32_t> rA = ranks(XA, true), rB = ranks(XB, false);
    // the cone: Cone^j = C_A^{j+1} + C_B^j, j = -1 .. qt - 1 (index j + 1)
    std::vector<uint32_t> cc;
    for (int jj = -1; jj < static_cast<int>(qt); ++jj) cc.push_back(XA.c[jj + 1] + (jj >= 0 ? XB.c[jj] : 0u));
    std::vector<Mat> Dc;   // Dc[j + 1]: D^j, j = -1 .. qt - 2
    for (int jj = -1; jj + 1 < static_cast<int>(qt); ++jj) {
        const uint32_t sA = XA.c[jj + 1], sB = jj >= 0 ? XB.c[jj] : 0u, tA = XA.c[jj + 2];
        Mat Dm(cc[jj + 2], cc[jj + 1]);
        Place(Dm, Neg(Z, XA.d[jj + 1]), 0, 0);
        Place(Dm, al[jj + 1], tA, 0);
        if (jj >= 0) Place(Dm, XB.d[jj], tA, sA);
        (void)sB;
        Dc.push_back(std::move(Dm));
    }
    std::vector<uint32_t> rc;   // rc[j + 1] = rank_Q D^j ((2)(d)(ii))
    {
        int64_t prev = 0;
        for (int jj = -1; jj + 1 < static_cast<int>(qt); ++jj) {
            const int64_t x = int64_t(cc[jj + 1]) - int64_t(bt.beta(jj)) - prev;
            ENGINE_CHECK_MSG(x >= 0, "AssemblePairMaps: negative cone Betti-recursion rank");
            rc.push_back(static_cast<uint32_t>(x));
            prev = x;
        }
    }
    // presentations
    L.ell = ell;
    L.qt = qt;
    for (uint32_t q = 0; q <= qt; ++q) {
        const Mat emptyA(XA.c[q], 0), emptyB(XB.c[q], 0), none(0, 0);
        L.HA.push_back(BuildPres(Z, q ? XA.d[q - 1] : emptyA, q ? rA[q - 1] : 0u, q < qt ? XA.d[q] : Mat(0, XA.c[q]), XA.c[q], bt.a(q)));
        L.HB.push_back(BuildPres(Z, q ? XB.d[q - 1] : emptyB, q ? rB[q - 1] : 0u, q < qt ? XB.d[q] : Mat(0, XB.c[q]), XB.c[q], bt.b(q)));
        // H^q(A, B) = H^{q-1}(Cone): Cone index j + 1 = q
        const int jj = static_cast<int>(q) - 1;
        const Mat prev = jj >= 0 ? Dc[jj] : Mat(cc[0], 0);
        const uint32_t rprev = jj >= 0 ? rc[jj] : 0u;
        const Mat next = (jj + 1 < static_cast<int>(qt)) ? Dc[jj + 1] : Mat(0, cc[q]);
        L.HAB.push_back(BuildPres(Z, prev, rprev, next, cc[q], bt.beta(jj)));
        rep.smith_forms = g_smith_forms;
        for (const Pres* P : {&L.HA.back(), &L.HB.back(), &L.HAB.back()}) {
            if (!P->ok) {
                rep.ok = false;
                rep.why = "presentation at q = " + std::to_string(q) + ": " + P->why;
                return false;
            }
            rep.min_prec = std::min(rep.min_prec, P->Z.k);
        }
    }
    // maps
    for (uint32_t q = 0; q <= qt; ++q) {
        L.res.push_back(InducedMap(L.HA[q], L.HB[q], al[q], q == 1 ? uint32_t(int64_t(bt.a(q)) + pc.opt->test_rank_offset) : bt.a(q)));
        L.cor.push_back(InducedMap(L.HB[q], L.HA[q], om[q], bt.a(q)));
        if (q < qt)   // j^q : C_B^q -> Cone^q = C_A^{q+1} + C_B^q (PAR-ASM: identity-like, InducedMapEmbed)
            L.j.push_back(InducedMapEmbed(L.HB[q], L.HAB[q + 1], XA.c[q + 1], 0, XB.c[q], bt.beta(static_cast<int>(q)), cc[q + 1], XB.c[q]));
        {   // dl^q : Cone^{q-1} = C_A^q + C_B^{q-1} -> C_A^q, (c, c') |-> c.  SIGN (D2 audit F8, [C, prop:descentmaps](2)(b)): dl = -delta,
            // delta the classical connecting map of 0 -> V -> Ind_B^A V -> Q -> 0 read through pi : (c, c') |-> [c'] ((2)(a)): a cone
            // cocycle has dc' = -alpha c, so delta [c'] = [-c].  Immaterial for groups, types, exactness and Hecke equivariance; it matters
            // only against an independently computed delta or through maps defined by delta (LES-1, Bockstein).
            L.dl.push_back(InducedMapEmbed(L.HAB[q], L.HA[q], 0, 0, XA.c[q], 0, XA.c[q], cc[q]));   // (PAR-ASM: identity-like)
        }
    }
    for (const auto* v : {&L.res, &L.cor, &L.j, &L.dl})
        for (const auto& m : *v) rep.min_prec = std::min(rep.min_prec, m.ZL.k ? m.ZL.k : m.Z.k);
    return CheckPiece(pc, L, Rr, rep, five);
}

bool CokernelPiece(const PairCtx& pc, uint32_t ell, uint32_t qt, Local& L, PieceReport& rep, std::string& five) {
    ENGINE_CHECK(pc.n % 2 == 0);
    const uint32_t K = ell == 3 ? LocalWidth(3) : integral_model::DefaultKBudget(ell);
    const U32Ctx ctx = U32Ctx::prime_power(ell, K);
    const ZMod Z(ell, K);
    rep.K = K;
    const Fibres F = BuildFibres(pc, ctx);
    const Betti& bt = pc.bt;
    const hecke::level::Gamma0LevelN A(pc.N), B(pc.MN);
    auto degree0 = [&](const hecke::level::Gamma0LevelN& G, uint32_t D) {
        const auto S = hecke::level::MonomialOf(G, hecke::Mat22::S(), pc.n, ctx).dense();
        const auto U = hecke::level::MonomialOf(G, hecke::Mat22::ST(), pc.n, ctx).dense();
        Mat N0(2 * D, D);
        for (uint32_t i = 0; i < D; ++i)
            for (uint32_t jj = 0; jj < D; ++jj) {
                N0.at(i, jj) = Z.sub(S[size_t(i) * D + jj], i == jj ? 1u : 0u);
                N0.at(D + i, jj) = Z.sub(U[size_t(i) * D + jj], i == jj ? 1u : 0u);
            }
        return N0;
    };
    const uint32_t DA = F.alpha.c, DB = F.alpha.r;
    const Mat GA = Transpose(FromDense(FixedGeneratorRows(pc.N, pc.n, ctx)));
    const Mat GB = Transpose(FromDense(FixedGeneratorRows(pc.MN, pc.n, ctx)));
    ENGINE_CHECK(GA.r == DA && GB.r == DB);
    Mat GBa(DB, GB.c + DA);
    Place(GBa, GB, 0, 0);
    Place(GBa, F.alpha, 0, GB.c);
    L.ell = ell;
    L.qt = qt;
    const uint32_t beta1 = bt.beta(1);
    for (uint32_t q = 0; q <= qt; ++q) {
        if (q == 0) {
            L.HA.push_back(BuildPres(Z, Mat(DA, 0), 0, degree0(A, DA), DA, bt.a(0)));
            L.HB.push_back(BuildPres(Z, Mat(DB, 0), 0, degree0(B, DB), DB, bt.b(0)));
        } else if (q == 1) {
            L.HA.push_back(BuildPres(Z, GA, DA - bt.a(1), Mat(0, DA), DA, bt.a(1)));
            L.HB.push_back(BuildPres(Z, GB, DB - bt.b(1), Mat(0, DB), DB, bt.b(1)));
        } else {
            L.HA.push_back(ZeroPres(Z, 0));
            L.HB.push_back(ZeroPres(Z, 0));
        }
        if (q == 2) L.HAB.push_back(BuildPres(Z, GBa, DB - beta1, Mat(0, DB), DB, beta1));
        else L.HAB.push_back(ZeroPres(Z, 0));
        for (const Pres* P : {&L.HA.back(), &L.HB.back(), &L.HAB.back()}) {
            if (!P->ok) {
                rep.ok = false;
                rep.why = "presentation at q = " + std::to_string(q) + ": " + P->why;
                return false;
            }
            rep.min_prec = std::min(rep.min_prec, P->Z.k);
        }
    }
    auto cmap = [&](const Pres& X, const Pres& Y, const Mat& f) -> Mat {
        if (X.c == f.c && Y.c == f.r) return f;
        return Mat(Y.c, X.c);   // a zero group at one end
    };
    Mat IdB = Identity(DB);
    for (uint32_t q = 0; q <= qt; ++q) {
        const uint32_t ra = q == 1 ? uint32_t(int64_t(bt.a(q)) + pc.opt->test_rank_offset) : bt.a(q);
        L.res.push_back(InducedMap(L.HA[q], L.HB[q], cmap(L.HA[q], L.HB[q], F.alpha), ra));
        L.cor.push_back(InducedMap(L.HB[q], L.HA[q], cmap(L.HB[q], L.HA[q], F.omega), bt.a(q)));
        if (q < qt) L.j.push_back(InducedMap(L.HB[q], L.HAB[q + 1], cmap(L.HB[q], L.HAB[q + 1], IdB), bt.beta(static_cast<int>(q))));
        L.dl.push_back(InducedMap(L.HAB[q], L.HA[q], Mat(L.HA[q].c, L.HAB[q].c), 0));
    }
    std::vector<std::vector<Mat>> Rr(F.Rs.size());
    for (size_t s = 0; s < F.Rs.size(); ++s)
        for (uint32_t q = 0; q <= qt; ++q) Rr[s].push_back(cmap(L.HB[q], L.HB[q], F.Rs[s]));
    for (const auto* v : {&L.res, &L.cor, &L.j, &L.dl})
        for (const auto& m : *v) rep.min_prec = std::min(rep.min_prec, m.ZL.k ? m.ZL.k : m.Z.k);
    rep.smith_forms = g_smith_forms;
    return CheckPiece(pc, L, Rr, rep, five);
}

uint32_t PsiOf(uint32_t N) { return PsiIndex(N); }

// |G| and |G^ab| from the multiplication table of A / B.
void GroupOrders(const les::QuotientGroup& Qg, uint32_t& order, uint32_t& ab) {
    order = Qg.index;
    const uint32_t k = Qg.index;
    // commutator subgroup: closure of {x y x^-1 y^-1}
    std::vector<uint32_t> invs(k, 0);
    uint32_t e = 0;
    for (uint32_t x = 0; x < k; ++x) {
        bool id = true;
        for (uint32_t y = 0; y < k; ++y)
            if (Qg.mult[x * k + y] != y) id = false;
        if (id) e = x;
    }
    for (uint32_t x = 0; x < k; ++x)
        for (uint32_t y = 0; y < k; ++y)
            if (Qg.mult[x * k + y] == e) invs[x] = y;
    std::set<uint32_t> C{e};
    for (uint32_t x = 0; x < k; ++x)
        for (uint32_t y = 0; y < k; ++y) C.insert(Qg.mult[Qg.mult[Qg.mult[x * k + y] * k + invs[x]] * k + invs[y]]);
    bool grew = true;
    while (grew) {
        grew = false;
        std::vector<uint32_t> cur(C.begin(), C.end());
        for (uint32_t a : cur)
            for (uint32_t b : cur)
                if (C.insert(Qg.mult[a * k + b]).second) grew = true;
    }
    ab = k / static_cast<uint32_t>(C.size());
}

}  // namespace

PairResult AssemblePairMaps(uint32_t N, uint32_t M, uint32_t n, const PairOptions& opt) {
    const auto t0 = clk::now();
    PairResult R;
    R.N = N;
    R.MN = N * M;
    R.n = n;
    ENGINE_CHECK_MSG(N >= 1 && M >= 2, "AssemblePairMaps: N >= 1, M >= 2");
    PairCtx pc;
    pc.N = N;
    pc.MN = N * M;
    pc.n = n;
    pc.opt = &opt;
    pc.index = PsiOf(N * M) / PsiOf(N);
    R.index = pc.index;
    {
        const hecke::level::Gamma0LevelN A(N), B(N * M);
        const auto pA = les::PermutationOf(A), pB = les::PermutationOf(B);
        const les::QuotientGroup Qg = les::QuotientOf(B, pB, A, pA);
        ENGINE_CHECK_MSG(Qg.contained && Qg.index == pc.index, "AssemblePairMaps: Gamma_0(MN) not of index psi(MN)/psi(N) in Gamma_0(N)");
        pc.normal = Qg.normal;
        if (Qg.normal) {
            for (uint32_t c : Qg.classes)
                if (c != 1) pc.sigmas.push_back(B.rep(c));
            GroupOrders(Qg, pc.G_order, pc.G_ab_order);
        }
    }
    R.normal = pc.normal;
    const LevelNCounts cA = LevelNClosedForm(N, n), cB = LevelNClosedForm(N * M, n);
    pc.bt.A[0] = cA.b0;
    pc.bt.A[1] = cA.b1;
    pc.bt.B[0] = cB.b0;
    pc.bt.B[1] = cB.b1;
    const bool even = n % 2 == 0;
    // S
    std::set<uint32_t> S{2};
    if (even) {
        S.insert(3);
        for (const auto& pr : BadPrimesN(N, n)) S.insert(pr.first);
        for (const auto& pr : BadPrimesN(N * M, n)) S.insert(pr.first);
        for (uint32_t p = 5; p <= pc.index; ++p)
            if (IsPrime(p) && pc.index % p == 0) S.insert(p);
    }
    R.primes.assign(S.begin(), S.end());
    std::ostringstream why;
    // rule 19: predict the dense pieces first, refuse above the ceiling before anything is built
    const uint32_t q2 = opt.q_top_two;
    ENGINE_CHECK_MSG(q2 >= 2 && q2 <= 6, "AssemblePairMaps: q_top_two in 2 .. 6");
    const bool dense3 = even && cA.nu3 > 0;
    const bool tree = DescentTreeEnabled();   // (D5) the tree route's own prediction
    auto pred = [&](uint32_t ell, uint32_t qt) { return tree ? PredictPairRssTree(ell, cA.D, cB.D, qt) : PredictPairRss(ell, cA.D, cB.D, qt); };
    R.rss_pred = pred(2, q2);
    if (dense3) R.rss_pred = std::max(R.rss_pred, pred(3, 6));
    // (D2 audit F3) refuse at 0.9 x the ceiling: the prediction covers the piece alone, not the process's base RSS nor D1's end-group
    // verification after the pieces
    if (R.rss_pred > opt.rss_cap_bytes / 10 * 9) {
        R.refused = true;
        R.refusal = "REFUSED (N, MN, n) = (" + std::to_string(N) + ", " + std::to_string(N * M) + ", " + std::to_string(n) +
                    "): the " + std::string(tree ? "tree" : "dense") + " pair piece predicts " + std::to_string(R.rss_pred >> 20) + " MiB above the ceiling " +
                    std::to_string(opt.rss_cap_bytes >> 20) + " MiB (D_A = " + std::to_string(cA.D) + ", D_B = " + std::to_string(cB.D) +
                    (tree ? ")" : "; the dense Iso^(2) route, HECKE_DESCENT_TREE=0)");
        R.t_total = since(t0);
        return R;
    }
    std::vector<Local> locals;
    std::vector<std::string> fives;
    for (uint32_t ell : R.primes) {
        PieceReport rep;
        rep.ell = ell;
        rep.min_prec = 1000;
        Local L;
        std::string five;
        const auto tp = clk::now();
        const uint32_t sm0 = g_smith_forms;
#if defined(__GLIBC__)
        malloc_trim(0);   // (D2 audit F3) return the freed heap of earlier pieces, so rss_base is what the process really retains
#endif
        rep.rss_base = CurrentRssBytes();
        RssSampler rs;
        bool ok = false;
        try {
            if (ell == 2) {
                rep.route = tree ? "Iso2T" : "Iso2";
                rep.K = LocalWidth(2);
                rep.q_top = q2;
                rep.rss_pred = pred(2, q2);
                ok = DensePiece(pc, 2, q2, L, rep, five);
            } else if (ell == 3 && dense3) {
                rep.route = tree ? "Iso3T" : "Iso3";
                rep.K = LocalWidth(3);
                rep.q_top = 6;
                rep.rss_pred = pred(3, 6);
                ok = DensePiece(pc, 3, 6, L, rep, five);
            } else {
                rep.route = ell == 3 ? "cokernel (nu_3 = 0, prop:descentNthree)" : "cokernel";
                rep.q_top = q2;
                ok = CokernelPiece(pc, ell, q2, L, rep, five);
            }
        } catch (const les::LevelPairRefusal& e) {
            rep.ok = false;
            rep.why = e.what();
        }
        rep.rss_meas = rs.stop();
        rep.seconds = since(tp);
        rep.smith_forms = g_smith_forms - sm0;
        if (!ok) why << " ell=" << ell << " (" << rep.route << "): " << rep.why << ";";
        if (!five.empty()) fives.push_back(five);
        R.exact_spots += rep.exact_spots;
        R.identity_checks += rep.identity_checks;
        R.pieces.push_back(rep);
        locals.push_back(std::move(L));
    }
    // glue ([C, prop:descentmaps](1)(c), (2)(c)): the free rank (equal in every piece) + the ell-primary parts
    if (why.str().empty()) {
        R.q_les = q2;
        auto at = [&](const Local& L, uint32_t q) { return q <= L.qt ? q : q - 2; };   // periodicity (2)(b), prop:descentN(5)
        auto glue = [&](auto getter, uint32_t q, AbelianGroupType& out, const char* what) {
            bool first = true;
            for (const Local& L : locals) {
                const AbelianGroupType g = getter(L, at(L, q));
                if (first) out.free_rank = g.free_rank;
                else if (g.free_rank != out.free_rank)
                    why << " free rank of " << what << " at q=" << q << " differs between the pieces;";
                first = false;
                for (const auto& [p, ex] : g.torsion)
                    for (uint32_t a : ex) out.add_torsion(p, a);
            }
        };
        R.HA.resize(6);
        R.HB.resize(6);
        R.HAB.resize(6);
        R.res.resize(6);
        R.cor.resize(6);
        R.j.resize(q2);
        R.dl.resize(q2 + 1);
        for (uint32_t q = 0; q <= 5; ++q) {
            glue([](const Local& L, uint32_t x) { return L.HA[x].type(); }, q, R.HA[q], "H(A)");
            glue([](const Local& L, uint32_t x) { return L.HB[x].type(); }, q, R.HB[q], "H(B)");
            glue([](const Local& L, uint32_t x) { return L.HAB[x].type(); }, q, R.HAB[q], "H(A,B)");
            glue([](const Local& L, uint32_t x) { return L.res[x].ker; }, q, R.res[q].ker, "ker res");
            glue([](const Local& L, uint32_t x) { return L.res[x].im; }, q, R.res[q].im, "im res");
            glue([](const Local& L, uint32_t x) { return L.res[x].coker; }, q, R.res[q].coker, "coker res");
            glue([](const Local& L, uint32_t x) { return L.cor[x].ker; }, q, R.cor[q].ker, "ker cor");
            glue([](const Local& L, uint32_t x) { return L.cor[x].im; }, q, R.cor[q].im, "im cor");
            glue([](const Local& L, uint32_t x) { return L.cor[x].coker; }, q, R.cor[q].coker, "coker cor");
        }
        for (uint32_t q = 0; q < q2; ++q) {
            glue([](const Local& L, uint32_t x) { return L.j[x].ker; }, q, R.j[q].ker, "ker j");
            glue([](const Local& L, uint32_t x) { return L.j[x].im; }, q, R.j[q].im, "im j");
            glue([](const Local& L, uint32_t x) { return L.j[x].coker; }, q, R.j[q].coker, "coker j");
        }
        for (uint32_t q = 0; q <= q2; ++q) {
            glue([](const Local& L, uint32_t x) { return L.dl[x].ker; }, q, R.dl[q].ker, "ker dl");
            glue([](const Local& L, uint32_t x) { return L.dl[x].im; }, q, R.dl[q].im, "im dl");
            glue([](const Local& L, uint32_t x) { return L.dl[x].coker; }, q, R.dl[q].coker, "coker dl");
        }
        // the rational parts against the closed forms (res injective and cor onto over Q; b_q(A, B) = beta_{q-1})
        for (uint32_t q = 0; q <= 5; ++q) {
            const Betti& b = pc.bt;
            if (R.HA[q].free_rank != b.a(q) || R.HB[q].free_rank != b.b(q) || R.HAB[q].free_rank != b.beta(int(q) - 1))
                why << " free ranks at q=" << q << " off the closed forms;";
            if (R.res[q].ker.free_rank != 0 || R.cor[q].coker.free_rank != 0 || R.res[q].coker.free_rank != b.b(q) - b.a(q))
                why << " rational ranks of res / cor at q=" << q << " off cor o res = [A:B];";
        }
        if (opt.verify_ends_with_d1) {
            const LevelNResult dA = AssembleDescentN(N, n), dB = AssembleDescentN(N * M, n);
            if (dA.refused || dB.refused) why << " the D1 end groups were refused;";
            else
                for (uint32_t q = 0; q <= 5; ++q)
                    if (dA.H[q] != R.HA[q] || dB.H[q] != R.HB[q]) why << " the end groups differ from D1's AssembleDescentN at q=" << q << ";";
        }
        R.five_term_checked = pc.normal && !fives.empty();
        for (const auto& f : fives) R.five_term += (R.five_term.empty() ? "" : "; ") + f;
    }
    R.t_total = since(t0);
    const std::string w = why.str();
    if (!w.empty()) {
        R.refused = true;
        R.refusal = "REFUSED (N, MN, n) = (" + std::to_string(N) + ", " + std::to_string(N * M) + ", " + std::to_string(n) + "):" + w;
        R.HA.clear();
        R.HB.clear();
        R.HAB.clear();
        R.res.clear();
        R.cor.clear();
        R.j.clear();
        R.dl.clear();
    }
    return R;
}

}  // namespace maps
}  // namespace integral_descent
