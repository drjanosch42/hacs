// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// DESCENT milestone D5c: the tree cusp pair as the route of the dense ell-local pair pieces, its comparisons with (Iso, Isob) and T_p by
// transport.  [C] = docs/latex/gl2_contract.tex (v1.23.0-pre.34, sec:treepair and its code map), cited by label.  See the header.
#include "DescentTreePair.h"
#include "DescentSmith.h"
#include "DescentParAsm.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <mutex>
#include <numeric>

#include "DescentLevelN.h"
#include "DescentTree.h"
#include "../Model/IntegralHecke.h"
#include "../../Hecke/ExactInt.h"
#include "../../../ModArithmetic/DenseElim.h"
#include "../../../ModArithmetic/DenseView.h"
#include "../../../ModArithmetic/EngineCheck.h"
#include "../../../ModArithmetic/ScalarMod.h"

namespace integral_descent {

// ---------------------------------------------------------------------------------------------------------------------------------
// the switch
// ---------------------------------------------------------------------------------------------------------------------------------
// (S24 CL-5) on EnvLever: "" = unset; an invalid value warns once and reads as the default (source=default).
namespace {
engine::lever::EnvLever g_pair("HECKE_DESCENT_TREE_PAIR",
                               engine::lever::Policy{true, engine::lever::Empty::Unset, engine::lever::Match::Exact01, engine::lever::Cache::Once, "WARNING: HECKE_DESCENT_TREE_PAIR=%s is neither 0 nor 1; the tree cusp pair stays on (default)\n", "default", "env", "default"});
bool pair_scoped() { return std::strcmp(g_pair.source(), "scoped") == 0; }
}  // namespace

const engine::lever::EnvLever& DescentTreePairLever() { return g_pair; }
bool DescentTreePairEnabled() {
    if (pair_scoped()) return g_pair.enabled();
    if (!DescentTreeEnabled()) return false;   // HECKE_DESCENT_TREE=0: the dense route everywhere
    return g_pair.env_enabled();
}
std::string DescentTreePairSource() {
    if (pair_scoped()) return "scoped";
    if (!DescentTreeEnabled()) return "tree-rollback";
    return g_pair.source();
}

namespace treepair {

using engine::DenseBuf;
using hecke::Mat22;
using integral_model::ImageSelection;
using integral_model::LChain;
using integral_model::QGroupRingElt;
using integral_model::Rat;
using integral_model::TreeModule;
using integral_model::TreePairFault;
namespace te = integral_model::tree_elt;
using clk = std::chrono::steady_clock;

namespace {

double since(clk::time_point t) { return std::chrono::duration<double>(clk::now() - t).count(); }

[[noreturn]] void Fail(const std::string& w) { throw hecke::HeckeInternalError(("[TreePairCertificate] " + w).c_str()); }

Mat FromRowMajor(uint32_t r, uint32_t c, std::vector<uint32_t>&& v) {
    Mat A;
    A.r = r;
    A.c = c;
    A.a = std::move(v);
    return A;
}
Mat Transpose(const Mat& A) {
    Mat B(A.c, A.r);
    for (uint32_t i = 0; i < A.r; ++i)
        for (uint32_t j = 0; j < A.c; ++j) B.at(j, i) = A.at(i, j);
    return B;
}
bool IsZero(const ZMod& Z, const Mat& A) {
    for (uint32_t x : A.a)
        if (Z.red(x)) return false;
    return true;
}
bool EqualMod(const ZMod& Z, const Mat& A, const Mat& B) {
    if (A.r != B.r || A.c != B.c) return false;
    for (size_t i = 0; i < A.a.size(); ++i)
        if (Z.red(A.a[i]) != Z.red(B.a[i])) return false;
    return true;
}
Mat Sub(const ZMod& Z, Mat A, const Mat& B) {
    ENGINE_CHECK(A.r == B.r && A.c == B.c);
    for (size_t i = 0; i < A.a.size(); ++i) A.a[i] = Z.sub(Z.red(A.a[i]), Z.red(B.a[i]));
    return A;
}
Mat Add(const ZMod& Z, Mat A, const Mat& B) {
    ENGINE_CHECK(A.r == B.r && A.c == B.c);
    for (size_t i = 0; i < A.a.size(); ++i) A.a[i] = Z.add(Z.red(A.a[i]), Z.red(B.a[i]));
    return A;
}
Mat Rows(const Mat& A, uint32_t r0, uint32_t n) {
    Mat B(n, A.c);
    if (n) std::copy(A.row(r0), A.row(r0) + size_t(n) * A.c, B.row(0));
    return B;
}
Mat Stack(const Mat& A, const Mat& B) {
    ENGINE_CHECK(A.c == B.c || A.r == 0 || B.r == 0);
    Mat C(A.r + B.r, A.r ? A.c : B.c);
    std::copy(A.a.begin(), A.a.end(), C.a.begin());
    std::copy(B.a.begin(), B.a.end(), C.a.begin() + A.a.size());
    return C;
}
Mat Inverse(const ZMod& Z, Mat G) {
    const uint32_t w = G.r;
    ENGINE_CHECK_MSG(G.r == G.c, "treepair: the block P[R, J] is not square");
    Mat I(w, w);
    for (uint32_t i = 0; i < w; ++i) I.at(i, i) = 1;
    for (uint32_t t = 0; t < w; ++t) {
        uint32_t p = t;
        while (p < w && Z.val(G.at(p, t)) != 0) ++p;
        ENGINE_CHECK_MSG(p < w, "treepair: the block P[R, J] is not invertible mod ell");
        if (p != t)
            for (uint32_t j = 0; j < w; ++j) {
                std::swap(G.at(t, j), G.at(p, j));
                std::swap(I.at(t, j), I.at(p, j));
            }
        const uint32_t u = Z.inv(G.at(t, t));
        for (uint32_t j = 0; j < w; ++j) {
            G.at(t, j) = Z.mul(G.at(t, j), u);
            I.at(t, j) = Z.mul(I.at(t, j), u);
        }
        for (uint32_t i = 0; i < w; ++i) {
            if (i == t || !G.at(i, t)) continue;
            const uint32_t f = G.at(i, t);
            for (uint32_t j = 0; j < w; ++j) {
                G.at(i, j) = Z.sub(G.at(i, j), Z.mul(f, G.at(t, j)));
                I.at(i, j) = Z.sub(I.at(i, j), Z.mul(f, I.at(t, j)));
            }
        }
    }
    return I;
}
// rows [o, o + w) of A replaced by Gi * (those rows)
void ApplyRowsInverse(const ZMod& Z, Mat& A, uint32_t o, const Mat& Gi) {
    const uint32_t w = Gi.r;
    if (!w || !A.c) return;
    Mat blk = Rows(A, o, w);
    const Mat nb = pair::MatMulZ(Z, Gi, blk);
    std::copy(nb.a.begin(), nb.a.end(), A.row(o));
}

// ---- rationals and the block-monomial application -------------------------------------------------------------------------------
uint64_t InvModU(uint64_t a, uint64_t m) {   // (S24 CL-G3, C-07) ScalarMod.h's extended Euclid
    uint64_t x = 0;
    ENGINE_CHECK_MSG(engine::scalar::try_invmod<uint64_t>(a, m, x), "treepair: a denominator is not a unit modulo ell^K");
    return x;
}
uint32_t ReduceRat(const U32Ctx& ctx, const Rat& r) {
    const uint64_t m = ctx.p;
    int64_t a = r.num % static_cast<int64_t>(m);
    if (a < 0) a += static_cast<int64_t>(m);
    return static_cast<uint32_t>(static_cast<uint64_t>(a) * InvModU(static_cast<uint64_t>(r.den), m) % m);
}
// out (rows of `out` block j, k columns) += coef * rho(g) in; blocks of d rows; row-major with k columns
void AddRhoApply(const U32Ctx& ctx, const integral_model::BlockMonomial<uint32_t>& B, uint32_t coef, const uint32_t* in, uint32_t* out,
                 uint32_t k) {
    const uint32_t d = B.d;
    const uint64_t M = ctx.p;
    for (uint32_t j = 0; j < B.m; ++j) {
        const std::vector<uint32_t>& blk = B.blocks[B.blk[j]];
        const uint32_t* src = in + size_t(B.col[j]) * d * k;
        uint32_t* dst = out + size_t(j) * d * k;
        for (uint32_t x = 0; x < d; ++x)
            for (uint32_t y = 0; y < d; ++y) {
                const uint64_t f = uint64_t(coef) * blk[size_t(x) * d + y] % M;
                if (!f) continue;
                const uint32_t* s = src + size_t(y) * k;
                uint32_t* t = dst + size_t(x) * k;
                for (uint32_t c = 0; c < k; ++c)
                    if (s[c]) t[c] = static_cast<uint32_t>((t[c] + f * s[c]) % M);
            }
    }
}
void AddQRhoApply(const U32Ctx& ctx, const TreeModule<uint32_t>& TM, const QGroupRingElt& a, const uint32_t* in, uint32_t* out, uint32_t k) {
    for (const auto& [g, r] : a.terms) {
        const uint32_t coef = ReduceRat(ctx, r);
        if (!coef) continue;
        AddRhoApply(ctx, TM.rho(integral_model::mat_of(g)), coef, in, out, k);
    }
}

// ---- the closed form of prop:treepair(2) ------------------------------------------------------------------------------------------
struct Sels {
    std::map<char, const ImageSelection*> sel;
    std::map<char, QGroupRingElt> e;
    std::map<char, Mat> Ginv;   // only for the selections that are not `all`
};

}  // namespace

pair::PairComplex BuildTreePairComplex(uint32_t ell, const U32Ctx& ctx, const TreeModule<uint32_t>& M, uint32_t T) {
    ENGINE_CHECK_MSG(ell == 2 || ell == 3, "BuildTreePairComplex: ell in {2, 3}");
    ENGINE_CHECK_MSG(T >= 1 && T <= 64, "BuildTreePairComplex: T");
    pair::PairComplex X;
    X.ell = ell;
    X.Z = ZMod(static_cast<uint32_t>(ctx.ell), ctx.is_field() ? 1u : ctx.k);
    X.T = T;
    X.route = "tree";
    const ZMod& Z = X.Z;
    const integral_model::TreeLocalForm<uint32_t> F = integral_model::MakeTreeLocalForm(ell, ctx, M, T);
    const uint32_t D = M.dim();
    const TreePairFault fault = integral_model::ActiveTreePairFault();   // None outside the validation's fault arm
    Sels S;
    if (ell == 2) {
        S.sel = {{'S', &F.sel_S}, {'U', &F.sel_U}, {'E', &F.sel_E}};
        S.e = {{'S', te::one()}, {'U', te::eU2()}, {'E', te::one()}};
    } else {
        S.sel = {{'S', &F.sel_S}, {'U', &F.sel_U}, {'E', &F.sel_E}};
        S.e = {{'S', te::eS4()}, {'U', te::ec()}, {'E', te::ec()}};
    }
    for (const auto& [x, sl] : S.sel) {
        if (sl->all || sl->rank() == 0) continue;
        const uint32_t r = sl->rank();
        std::vector<uint32_t> g(size_t(r) * r, 0);
        integral_model::AddRhoBlock(ctx, g, r, 0, 0, M, S.e.at(x), *sl, *sl);   // G = rho(e)[R, J]
        S.Ginv.emplace(x, Inverse(Z, FromRowMajor(r, r, std::move(g))));
    }
    auto cols_of = [&](uint32_t q) -> std::vector<char> {
        if (ell == 2) return q == 0 ? std::vector<char>{'S', 'U'} : std::vector<char>{'S', 'U', 'E'};
        return q == 0 ? std::vector<char>{'S', 'U'} : (q == 1 ? std::vector<char>{'U', 'E'} : std::vector<char>{'U'});
    };
    // the cochain ranks (the column layout of prop:isomodeltree(2)(e))
    for (uint32_t q = 0; q <= T; ++q) {
        uint32_t w = 0;
        for (char x : cols_of(q)) w += S.sel.at(x)->rank();
        ENGINE_CHECK_MSG(w == F.cochain_rank(q), "BuildTreePairComplex: the column layout disagrees with prop:isomodeltree(2)(e)");
        X.c.push_back(w);
    }
    // the cusp model: B^q = M (+) M (ell = 2; M for q = 0), M^{-I} in degrees 0, 1 (ell = 3); every column the edge's selection
    const ImageSelection& sE = *S.sel.at('E');
    const uint32_t rE = sE.rank();
    for (uint32_t q = 0; q <= T; ++q) X.cb.push_back(ell == 2 ? (q == 0 ? D : 2 * D) : (q <= 1 ? rE : 0u));
    const QGroupRingElt I = te::one(), Cg = te::c(), eE = S.e.at('E');
    const QGroupRingElt Tm = te::g(fault == TreePairFault::CuspOneMinusT2 ? Mat22::T() * Mat22::T() : Mat22::T());
    const QGroupRingElt one_minus_T = te::sum(I, Tm, -1) * eE;
    auto eps = [&](uint32_t t) { return (t % 2 == 1 ? te::sum(I, Cg, -1) : te::sum(I, Cg, 1)) * eE; };
    const Mat22 si = fault == TreePairFault::AxisSForSinv ? Mat22::S() : Mat22::S().inv_unimodular();   // VALIDATION ONLY
    const QGroupRingElt SI = te::g(si);
    QGroupRingElt gam = SI;
    gam.add(te::g(Mat22::T()), Rat::of(-1));   // S^{-1} - T
    const bool drop_beta = fault == TreePairFault::AxisDropBeta;
    auto finish_rows = [&](Mat& A) {   // J-coordinates on the cusp rows (all rows are edge-type columns)
        auto it = S.Ginv.find('E');
        if (it == S.Ginv.end() || !A.r) return;
        for (uint32_t o = 0; o < A.r; o += rE) ApplyRowsInverse(Z, A, o, it->second);
    };
    // the bulk: d^q = G^{-1} pi_R d B_J (the closed form of prop:isomodeltree(2), (2)(d)'s periodic copies)
    auto build_d = [&](uint32_t q) {
        Mat A = FromRowMajor(X.c[q + 1], X.c[q], F.differential(ctx, M, q));
        uint32_t o = 0;
        for (char y : cols_of(q + 1)) {
            const uint32_t w = S.sel.at(y)->rank();
            auto it = S.Ginv.find(y);
            if (it != S.Ginv.end()) ApplyRowsInverse(Z, A, o, it->second);
            o += w;
        }
        return A;
    };
    auto build_db = [&](uint32_t q) {
        const uint32_t rows = X.cb[q + 1], cols = X.cb[q];
        std::vector<uint32_t> v(size_t(rows) * cols, 0);
        auto add = [&](uint32_t r0, uint32_t c0, const QGroupRingElt& a) { integral_model::AddRhoBlock(ctx, v, cols, r0, c0, M, a, sE, sE); };
        if (rows && cols) {
            if (ell == 2) {
                if (q == 0) {
                    add(0, 0, eps(1));          // e_1 : (1 - c) u
                    add(D, 0, one_minus_T);     // z_0 : (1 - T) u
                } else {
                    add(0, 0, eps(q + 1));      // e_{q+1} : eps_{q+1} u
                    add(D, 0, one_minus_T);     // z_q : (1 - T) u
                    add(D, D, te::sum(QGroupRingElt{}, eps(q), -1));   //      - eps_q w
                }
            } else if (q == 0) {
                add(0, 0, one_minus_T);         // z_0 : (1 - T) u on M^{-I}
            }
        }
        Mat A = FromRowMajor(rows, cols, std::move(v));
        finish_rows(A);
        return A;
    };
    // the axis map a^* : C^q_T -> B^q (prop:treepair(2)(a), (b))
    auto build_r = [&](uint32_t q) {
        const uint32_t rows = X.cb[q], cols = X.c[q];
        std::vector<uint32_t> v(size_t(rows) * cols, 0);
        auto add = [&](uint32_t r0, uint32_t c0, const QGroupRingElt& a, char col) {
            integral_model::AddRhoBlock(ctx, v, cols, r0, c0, M, a, sE, *S.sel.at(col));
        };
        if (rows && cols) {
            if (ell == 2) {
                const uint32_t rU = S.sel.at('U')->rank();
                if (q == 0) {
                    add(0, 0, I, 'S');                                                   // u = x
                } else {
                    add(0, 0, q % 2 == 0 ? I : te::sum(I, te::S()), 'S');               // u = a_q x
                    if (q % 2 == 1) {
                        add(D, 0, te::sum(QGroupRingElt{}, SI, -1), 'S');               // w: - S^{-1} x
                        if (!drop_beta) add(D, D, SI * te::eU2(), 'U');                  //    + S^{-1} y
                    }
                    add(D, D + rU, gam, 'E');                                           //    + (S^{-1} - T) z
                }
            } else {
                if (q == 0) {
                    add(0, 0, te::eS4(), 'S');                                          // u = x (x in M^<S>)
                } else if (q == 1) {
                    if (!drop_beta) add(0, 0, SI * te::ec(), 'U');                       // w = S^{-1} y
                    add(0, S.sel.at('U')->rank(), gam * te::ec(), 'E');                 //   + (S^{-1} - T) z
                }
            }
        }
        Mat A = FromRowMajor(rows, cols, std::move(v));
        finish_rows(A);
        return A;
    };
    // The periodic copies point at their ROOT (an index built directly): d^q = d^{rep(q)}; d_B^q = d_B^{q-2}, a^{*,q} = a^{*,q-2} at
    // ell = 2, q >= 3 (rep and q - 2 - 2 ... land in {1, 2}).  (PAR-ASM, lever HECKE_DESCENT_PAR_ASM) every root matrix is an
    // independent task (its own slot; the shared module's rho memo is mutex-guarded), then every copy is a task; in the serial order
    // (bulk, cusp, axis) when the lever is off.  An exception re-throws from the lowest task index, i.e. the one the serial loops
    // would have hit first.  The matrices are the same values in the same slots.
    auto root2 = [&](uint32_t q) {
        while (ell == 2 && q >= 3) q -= 2;
        return q;
    };
    X.d.assign(T, Mat());
    X.db.assign(T, Mat());
    X.r.assign(T + 1, Mat());
    struct Job {
        char kind;   // 'd', 'b', 'r'
        uint32_t q;
    };
    std::vector<Job> roots, copies;
    for (uint32_t q = 0; q < T; ++q) (F.representative(q) < q ? copies : roots).push_back({'d', q});
    for (uint32_t q = 0; q < T; ++q) (root2(q) < q ? copies : roots).push_back({'b', q});
    for (uint32_t q = 0; q <= T; ++q) (root2(q) < q ? copies : roots).push_back({'r', q});
    const uint32_t wk = par::Enabled(par::Lever::Asm) ? par::Threads() : 1u;
    par::ForEachIndex(roots.size(), wk, [&](size_t i) {
        const Job& j = roots[i];
        if (j.kind == 'd') X.d[j.q] = build_d(j.q);
        else if (j.kind == 'b') X.db[j.q] = build_db(j.q);
        else X.r[j.q] = build_r(j.q);
    });
    par::ForEachIndex(copies.size(), wk, [&](size_t i) {
        const Job& j = copies[i];
        if (j.kind == 'd') X.d[j.q] = X.d[F.representative(j.q)];
        else if (j.kind == 'b') X.db[j.q] = X.db[root2(j.q)];
        else X.r[j.q] = X.r[root2(j.q)];
    });
    // certificates of the constructed maps (prop:treepair(6)): d_B o d_B = 0 and d_B a^* = a^* d_T, entrywise (sparse left factors)
    // (PAR-ASM, lever HECKE_DESCENT_PAR_ASM) the same two identities streamed in row blocks and threaded, no product materialised
    // (maps::ProductsEqual, PAR-D2; the operands are reduced: AddRhoBlock and MatMulZ return reduced representatives).  The same
    // booleans: a product is 0 iff it equals the product with a zero factor; EqualMod compares reduced entries, as ProductsEqual does.
    const bool par_cert = par::Enabled(par::Lever::Asm);
    for (uint32_t q = 0; q + 1 < T; ++q) {
        const bool zero = par_cert ? maps::ProductsEqual(Z, X.db[q + 1], X.db[q], Mat(X.db[q + 1].r, 1), Mat(1, X.db[q].c))
                                   : IsZero(Z, pair::MatMulZ(Z, X.db[q + 1], X.db[q]));
        if (!zero) Fail("d_B o d_B != 0 on the cusp cochains at q = " + std::to_string(q) + " (ell = " + std::to_string(ell) + ")");
    }
    for (uint32_t q = 0; q < T; ++q) {
        const bool same = par_cert ? maps::ProductsEqual(Z, X.db[q], X.r[q], X.r[q + 1], X.d[q])
                                   : EqualMod(Z, pair::MatMulZ(Z, X.db[q], X.r[q]), pair::MatMulZ(Z, X.r[q + 1], X.d[q]));
        if (!same) Fail("a^* is not a cochain map (d_B a^* != a^* d_T) at q = " + std::to_string(q) + " (ell = " + std::to_string(ell) + ")");
    }
    return X;
}

std::vector<uint32_t> TreeConeRanksModPrime(uint32_t ell, uint32_t N, uint32_t n, uint32_t ellp, uint32_t T) {
    const U32Ctx F(ellp);
    const TreeModule<uint32_t> M = integral_model::TreeInducedSymModule(N, n, F);
    const pair::PairComplex X = BuildTreePairComplex(ell, F, M, T);
    std::vector<uint32_t> ranks;
    for (int j = -1; j + 2 <= static_cast<int>(T); ++j) {
        Mat Dm = pair::ConeDifferential(X, j);
        if (!Dm.r || !Dm.c) {
            ranks.push_back(0);
            continue;
        }
        DenseBuf<uint32_t> W = DenseBuf<uint32_t>::adopt(std::move(Dm.a), Dm.r, Dm.c);
        ranks.push_back(engine::elim::rref(F, engine::view(W)).rank);
    }
    return ranks;
}

uint64_t TreePairRssStructural(uint32_t ell, uint32_t D, uint32_t T) {
    ENGINE_CHECK(ell == 2 || ell == 3);
    // reduced ranks per unit of D (prop:treepair(2)(d); ell = 3 for n even: M^{-I} = M, M^<S> ~ D / 2)
    auto c = [&](uint32_t q) -> uint64_t {
        const double w = ell == 2 ? (q == 0 ? 4.0 / 3 : 7.0 / 3) : (q == 0 ? 1.5 : (q == 1 ? 2.0 : 1.0));
        return uint64_t(w * D) + 1;
    };
    auto cb = [&](uint32_t q) -> uint64_t { return ell == 2 ? (q == 0 ? D : 2ull * D) : (q <= 1 ? D : 0ull); };
    uint64_t stored = 0, cmax = 0;
    for (uint32_t q = 0; q < T; ++q) stored += c(q + 1) * c(q) + cb(q + 1) * cb(q) + cb(q) * c(q);
    stored += cb(T) * c(T);
    for (int j = -1; j + 2 <= static_cast<int>(T); ++j) {
        const uint64_t cj = c(static_cast<uint32_t>(j + 1)) + (j >= 0 ? cb(static_cast<uint32_t>(j)) : 0);
        const uint64_t cj1 = c(static_cast<uint32_t>(j + 2)) + cb(static_cast<uint32_t>(j + 1));
        cmax = std::max(cmax, std::max(cj, cj1));
    }
    // two cone differentials alive (RunPairLes, D5c), the largest Smith form with left transforms (A, L, L^{-1}T) and the right-transform
    // Smith of the next differential on the complement (two more c x c): 7 cmax^2 words.  (PAR-D2, the lever HECKE_DESCENT_SMITH_BLOCKED
    // on: the Smith form runs in the consumed D^{q-2} and keeps its transforms factored, E = the complement columns of D^{q-1} is consumed
    // by its own factorisation: D^{q-2} (= the factor), D^{q-1}, E: 3 cmax^2 words; PARD2_report.md Sec. 5.)
    const uint64_t smith_words = maps::DescentSmithBlockedEnabled() ? 3 : 7;
    return 4ull * (stored + smith_words * cmax * cmax);
}
uint64_t PredictTreePairRss(uint32_t ell, uint32_t D, uint32_t T) {
    // calibration factor: DESCENT5c_predictions.md (filed before the measuring runs)
    return TreePairRssStructural(ell, D, T) / 10 * 12 + (256ull << 20);
}

// ---------------------------------------------------------------------------------------------------------------------------------
// the shared chain-level comparison
// ---------------------------------------------------------------------------------------------------------------------------------
const integral_model::TreePairComparison& SharedTreePairComparison(uint32_t ell) {
    static std::mutex mu;
    static std::map<std::pair<uint32_t, uint32_t>, std::unique_ptr<integral_model::TreePairComparison>> memo;
    const uint32_t f = static_cast<uint32_t>(integral_model::ActiveTreePairFault());
    std::lock_guard<std::mutex> lk(mu);
    auto& slot = memo[{ell, f}];
    if (!slot) {
        const integral_model::IntegralTreeComparison& cmp = hecke3::SharedTreeComparison();
        std::unique_ptr<integral_model::TreePairComparison> c;
        try {
            c = std::make_unique<integral_model::TreePairComparison>(cmp, ell, hecke3::kTreeComparisonTop);
        } catch (const hecke::HeckeOverflow&) {
            throw;
        } catch (const std::logic_error& e) {
            // a solver's precondition failed while (L) / (L') were built: the inputs are not what (3)(b), (c) say -- a defect
            Fail(std::string("the comparison of prop:treepair(3) could not be built at ell = ") + std::to_string(ell) + ": " + e.what());
        }
        const integral_model::TreePairCertificates cert = c->certify();
        if (!cert.all()) Fail("prop:treepair(1)(b), (3) not certified at ell = " + std::to_string(ell) + ": " + cert.first_failure);
        slot = std::move(c);
    }
    return *slot;
}

// ---------------------------------------------------------------------------------------------------------------------------------
// TreePairCochains
// ---------------------------------------------------------------------------------------------------------------------------------
TreePairCochains::~TreePairCochains() = default;

TreePairCochains::TreePairCochains(uint32_t ell, uint32_t N, uint32_t n, uint32_t T, uint32_t K)
    : ell_(ell), N_(N), n_(n), T_(T) {
    const auto t0 = clk::now();
    ENGINE_CHECK_MSG(ell == 2 || ell == 3, "TreePairCochains: ell in {2, 3}");
    ENGINE_CHECK_MSG(T >= 2 && T <= hecke3::kTreeComparisonTop + 1, "TreePairCochains: T in [2, 6]");
    const uint32_t W = K ? K : LocalWidth(ell);
    Z_ = ZMod(ell, W);
    ctx_ = U32Ctx::prime_power(ell, W);
    TM_ = integral_model::TreeInducedSymModule(N, n, ctx_);
    D_ = TM_.dim();
    X_ = BuildTreePairComplex(ell, ctx_, TM_, T);
    cmp_ = &SharedTreePairComparison(ell);
    tt_ = std::make_unique<hecke3::TreeTransport>(ell, N, n, std::min(T, hecke3::kTreeComparisonTop), W);
    for (uint32_t q = 0; q <= std::min(T, hecke3::kTreeComparisonTop + 1); ++q)
        ENGINE_CHECK_MSG(tt_->rank(q) == X_.c[q], "TreePairCochains: the tree's J-coordinates differ from TreeTransport's");
    const auto& iso = cmp_->iso();
    bpos_.assign(T + 1, {});
    for (uint32_t q = 0; q <= T; ++q) {
        bpos_[q].assign(iso.rank(q), -1);
        int32_t b = 0;
        for (uint32_t c = 0; c < iso.rank(q); ++c)
            if (iso.cells(q)[c].boundary) bpos_[q][c] = b++;
    }
    for (uint32_t q = 0; q <= T; ++q)
        ENGINE_CHECK_MSG(X_.cb[q] == cmp_->cusp_local().rank(q) * D_,
                         "TreePairCochains: the cusp cochains are not the full module on every cell (at ell = 3: n odd, M^{-I} = 0; the "
                         "pair's 3-adic piece runs for n even only)");
    build_seconds = since(t0);
}

uint32_t TreePairCochains::isob_rank(uint32_t q) const {
    uint32_t b = 0;
    for (int32_t x : bpos_.at(q)) b += x >= 0 ? 1u : 0u;
    return b * D_;
}

uint64_t TreePairCochains::transport_bytes(uint32_t q_max, uint32_t cols) const {
    uint64_t worst = 0;
    for (uint32_t a = 1; a <= std::min(q_max, std::min(T_, hecke3::kTreeComparisonTop)); ++a) {
        const uint64_t w = uint64_t(tt_->iso_full(a)) + tt_->full(a) + isob_rank(a - 1) + X_.c[a] + X_.cb[a - 1];
        worst = std::max(worst, w);
    }
    // live at the peak (inside apply_hecke on the checked alphas): j_Phi's output and its stacked copy, the bulk rows cut out, T~ of
    // them, the stacked T~Y and j_Psi's result -- six column blocks of the widest width (the alphas, T'_T alpha and eta alpha are smaller)
    return 4ull * 6 * worst * cols;
}

const pair::PairComplex& TreePairCochains::iso() const {
    if (!iso_) iso_ = std::make_unique<pair::PairComplex>(pair::BuildFreePairN(N_, n_, ctx_, T_));
    return *iso_;
}

Mat TreePairCochains::phi_d_star(uint32_t q, const Mat& C) const {
    const auto& Bl = cmp_->cusp_local();
    ENGINE_CHECK_MSG(C.r == Bl.rank(q) * D_, "phi_d_star: the input is not on B^q");
    const uint32_t k = C.c;
    Mat out(isob_rank(q), k);
    const auto& val = cmp_->phi_d().value.at(q);
    for (uint32_t y = 0; y < val.size(); ++y) {
        if (bpos_[q][y] < 0) continue;
        for (const auto& [x, b] : val[y])
            AddQRhoApply(ctx_, TM_, b, C.a.data() + size_t(x) * D_ * k, out.a.data() + size_t(bpos_[q][y]) * D_ * k, k);
    }
    return out;
}
Mat TreePairCochains::psi_d_star(uint32_t q, const Mat& C) const {
    const auto& Bl = cmp_->cusp_local();
    ENGINE_CHECK_MSG(C.r == isob_rank(q), "psi_d_star: the input is not on Isob^q");
    const uint32_t k = C.c;
    Mat out(Bl.rank(q) * D_, k);
    const auto& val = cmp_->psi_d().value.at(q);
    for (uint32_t u = 0; u < val.size(); ++u)
        for (const auto& [x, b] : val[u]) {
            const int32_t bx = bpos_[q].at(x);
            ENGINE_CHECK_MSG(bx >= 0, "psi_d_star: Psi_d leaves Isob");
            AddQRhoApply(ctx_, TM_, b, C.a.data() + size_t(bx) * D_ * k, out.a.data() + size_t(u) * D_ * k, k);
        }
    return out;
}
Mat TreePairCochains::y_phi_star(uint32_t q, const Mat& A) const {
    ENGINE_CHECK_MSG(A.r == tt_->full(q + 1), "y_phi_star: the input is not on C^{q+1}_T (full)");
    const uint32_t k = A.c;
    Mat out(isob_rank(q), k);
    const auto& val = cmp_->Y_phi().value.at(q);
    for (uint32_t y = 0; y < val.size(); ++y) {
        if (bpos_[q][y] < 0) continue;
        for (const auto& [x, b] : val[y])
            AddQRhoApply(ctx_, TM_, b, A.a.data() + size_t(x) * D_ * k, out.a.data() + size_t(bpos_[q][y]) * D_ * k, k);
    }
    return out;
}
namespace {
// splitmix64: the deterministic stream of the random vectors (seeded by the cell and the cone degree; never by time)
struct SplitMix {
    uint64_t x;
    uint64_t next() {
        uint64_t z = (x += 0x9e3779b97f4a7c15ull);
        z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ull;
        z = (z ^ (z >> 27)) * 0x94d049bb133111ebull;
        return z ^ (z >> 31);
    }
    uint32_t uniform(uint64_t m) {   // uniform in [0, m) by rejection
        const uint64_t lim = UINT64_MAX - UINT64_MAX % m;
        uint64_t r;
        do r = next(); while (r >= lim);
        return static_cast<uint32_t>(r % m);
    }
};
Mat RandomMat(const ZMod& Z, uint32_t r, uint32_t c, uint64_t seed) {
    SplitMix g{seed};
    Mat M(r, c);
    for (auto& v : M.a) v = g.uniform(Z.m);
    return M;
}
}  // namespace

namespace {
struct PlantedCocycleEta {
    uint32_t q = 0;
    std::vector<uint32_t> z;
    uint64_t seed = 0;
};
std::atomic<const PlantedCocycleEta*> g_planted{nullptr};
}  // namespace
ScopedPlantedCocycleEta::ScopedPlantedCocycleEta(uint32_t q, std::vector<uint32_t> z, uint64_t seed) {
    g_planted.store(new PlantedCocycleEta{q, std::move(z), seed});
}
ScopedPlantedCocycleEta::~ScopedPlantedCocycleEta() { delete g_planted.exchange(nullptr); }

Mat TreePairCochains::y_psi_star(uint32_t q, const Mat& A) const {
    const auto& Bl = cmp_->cusp_local();
    ENGINE_CHECK_MSG(A.r == tt_->iso_full(q + 1), "y_psi_star: the input is not on C^{q+1}_I");
    const uint32_t k = A.c;
    Mat out(Bl.rank(q) * D_, k);
    const auto& val = cmp_->Y_psi().value.at(q);
    for (uint32_t u = 0; u < val.size(); ++u)
        for (const auto& [x, b] : val[u])
            AddQRhoApply(ctx_, TM_, b, A.a.data() + size_t(x) * D_ * k, out.a.data() + size_t(u) * D_ * k, k);
    // VALIDATION ONLY: the planted shared-primitive error alpha |-> (r . alpha) z (ScopedPlantedCocycleEta)
    if (const PlantedCocycleEta* P = g_planted.load(); P && P->q == q && P->z.size() == out.r) {
        const Mat r = RandomMat(Z_, 1, A.r, P->seed);
        for (uint32_t c = 0; c < k; ++c) {
            uint32_t sc = 0;
            for (uint32_t i = 0; i < A.r; ++i) sc = Z_.add(sc, Z_.mul(r.a[i], Z_.red(A.at(i, c))));
            for (uint32_t i = 0; i < out.r; ++i) out.at(i, c) = Z_.add(out.at(i, c), Z_.mul(sc, Z_.red(P->z[i])));
        }
    }
    return out;
}

namespace {
Mat ToMat(uint32_t r, uint32_t c, std::vector<uint32_t>&& v) { return FromRowMajor(r, c, std::move(v)); }
}  // namespace

Mat TreePairCochains::phi_T_star(uint32_t q, const Mat& U) const {
    std::vector<uint32_t> V, W;
    tt_->to_full(q, U.a, U.c, V);
    tt_->phi_star(q, V, U.c, W);
    return ToMat(tt_->iso_full(q), U.c, std::move(W));
}
Mat TreePairCochains::psi_T_star(uint32_t q, const Mat& A) const {
    std::vector<uint32_t> V, U;
    tt_->psi_star(q, A.a, A.c, V);
    tt_->from_full(q, V, A.c, U);
    return ToMat(X_.c[q], A.c, std::move(U));
}
Mat TreePairCochains::axis_star_chain(uint32_t q, const Mat& U) const {
    const auto& Bl = cmp_->cusp_local();
    const uint32_t k = U.c;
    std::vector<uint32_t> V;
    tt_->to_full(q, U.a, k, V);
    Mat out(Bl.rank(q) * D_, k);
    const auto& val = cmp_->axis().value.at(q);
    for (uint32_t u = 0; u < val.size(); ++u)
        for (const auto& [x, b] : val[u]) AddQRhoApply(ctx_, TM_, b, V.data() + size_t(x) * D_ * k, out.a.data() + size_t(u) * D_ * k, k);
    return out;
}
Mat TreePairCochains::cusp_d_chain(uint32_t q, const Mat& C) const {
    const auto& Bl = cmp_->cusp_local();
    const uint32_t k = C.c;
    ENGINE_CHECK_MSG(C.r == Bl.rank(q) * D_ && q + 1 <= Bl.q_top(), "cusp_d_chain: shapes");
    Mat out(Bl.rank(q + 1) * D_, k);
    // (d c)(y) = sum over the entries of d(y) = sum coeff e_row of rho(coeff) c(row)
    for (const integral_model::LEntry& e : Bl.differential(q + 1))
        AddQRhoApply(ctx_, TM_, e.coeff, C.a.data() + size_t(e.row) * D_ * k, out.a.data() + size_t(e.col) * D_ * k, k);
    return out;
}

Mat TreePairCochains::j_phi(int j, const Mat& X) const {
    ENGINE_CHECK_MSG(j >= -1 && j + 1 <= static_cast<int>(std::min(T_, hecke3::kTreeComparisonTop)), "j_phi: cone degree out of range");
    const uint32_t a = static_cast<uint32_t>(j + 1);
    const uint32_t ca = X_.c[a], cbj = j >= 0 ? X_.cb[static_cast<uint32_t>(j)] : 0u;
    ENGINE_CHECK_MSG(X.r == ca + cbj, "j_phi: the input is not on Cone(a^*)^j");
    const uint32_t k = X.c;
    std::vector<uint32_t> af, first;
    tt_->to_full(a, Rows(X, 0, ca).a, k, af);
    tt_->phi_star(a, af, k, first);
    Mat out = ToMat(tt_->iso_full(a), k, std::move(first));
    if (j < 0) return out;
    const uint32_t q = static_cast<uint32_t>(j);
    Mat second = phi_d_star(q, Rows(X, ca, cbj));
    const Mat Y = y_phi_star(q, ToMat(tt_->full(a), k, std::move(af)));
    second = integral_model::ActiveTreePairFault() == TreePairFault::JPhiPlusY ? Add(Z_, std::move(second), Y) : Sub(Z_, std::move(second), Y);
    return Stack(out, second);
}

Mat TreePairCochains::j_psi(int j, const Mat& Y) const {
    ENGINE_CHECK_MSG(j >= -1 && j + 1 <= static_cast<int>(std::min(T_, hecke3::kTreeComparisonTop)), "j_psi: cone degree out of range");
    const uint32_t a = static_cast<uint32_t>(j + 1);
    const uint32_t ia = tt_->iso_full(a), ib = j >= 0 ? isob_rank(static_cast<uint32_t>(j)) : 0u;
    ENGINE_CHECK_MSG(Y.r == ia + ib, "j_psi: the input is not on Cone(rres_Iso)^j");
    const uint32_t k = Y.c;
    const Mat alpha = Rows(Y, 0, ia);
    std::vector<uint32_t> v, u;
    tt_->psi_star(a, alpha.a, k, v);
    tt_->from_full(a, v, k, u);
    Mat out = ToMat(X_.c[a], k, std::move(u));
    if (j < 0) return out;
    const uint32_t q = static_cast<uint32_t>(j);
    const Mat second = Sub(Z_, psi_d_star(q, Rows(Y, ia, ib)), y_psi_star(q, alpha));
    return Stack(out, second);
}

Mat TreePairCochains::apply_hecke(const IsoCochainOp& T, int j, const Mat& X, bool& refused_closure) const {
    refused_closure = false;
    const uint32_t a = static_cast<uint32_t>(j + 1);
    const uint32_t k = X.c;
    auto composite = [&](const Mat& In) {
        const Mat Y = j_phi(j, In);
        const uint32_t ia = tt_->iso_full(a);
        std::vector<uint32_t> tb;
        T.apply(a, Rows(Y, 0, ia).a, k, tb);
        Mat TY = ToMat(ia, k, std::move(tb));
        if (j >= 0) {
            const uint32_t q = static_cast<uint32_t>(j);
            std::vector<uint32_t> td;
            if (!T.apply_boundary(q, bpos_[q], Rows(Y, ia, isob_rank(q)).a, k, td)) refused_closure = true;
            TY = Stack(TY, ToMat(isob_rank(q), k, std::move(td)));
        }
        return j_psi(j, TY);
    };
    if (integral_model::ActiveTreePairFault() != TreePairFault::EtaDropped || j < 0) return composite(X);
    // VALIDATION ONLY: T^c without eta_T -- the diagonal (T'_T alpha, T'_d c)
    const uint32_t ca = X_.c[a];
    Mat Xa = X, Xc = X;
    for (uint32_t i = ca; i < X.r; ++i) std::fill(Xa.row(i), Xa.row(i) + k, 0u);
    for (uint32_t i = 0; i < ca; ++i) std::fill(Xc.row(i), Xc.row(i) + k, 0u);
    const Mat Ra = composite(Xa), Rc = composite(Xc);
    Mat out = Rc;
    for (uint32_t i = 0; i < ca; ++i)
        for (uint32_t c = 0; c < k; ++c) out.at(i, c) = Z_.add(Ra.at(i, c), Rc.at(i, c));
    return out;
}

Mat TreePairCochains::bulk_hecke(const IsoCochainOp& T, uint32_t q, const Mat& X) const {
    // T'_T = Psi_T^* T_b Phi_T^* (prop:isomodeltree(6); D3's TreeTransport maps)
    const uint32_t k = X.c;
    std::vector<uint32_t> V, W, W2, V2, out;
    tt_->to_full(q, X.a, k, V);
    tt_->phi_star(q, V, k, W);
    V.clear();
    V.shrink_to_fit();
    T.apply(q, W, k, W2);
    W.clear();
    W.shrink_to_fit();
    tt_->psi_star(q, W2, k, V2);
    tt_->from_full(q, V2, k, out);
    return ToMat(X_.c[q], k, std::move(out));
}

Mat TreePairCochains::bdry_hecke(const IsoCochainOp& T, uint32_t q, const Mat& X, bool& closed) const {
    const Mat P = phi_d_star(q, X);
    std::vector<uint32_t> td;
    closed = T.apply_boundary(q, bpos_[q], P.a, X.c, td);
    return psi_d_star(q, ToMat(isob_rank(q), X.c, std::move(td)));
}

Mat TreePairCochains::eta(const IsoCochainOp& T, int j, const Mat& A, bool& closed) const {
    ENGINE_CHECK_MSG(j >= 0, "eta: cone degree >= 0");
    const uint32_t a = static_cast<uint32_t>(j + 1), q = static_cast<uint32_t>(j);
    const uint32_t k = A.c;
    std::vector<uint32_t> af, phi, tb;
    tt_->to_full(a, A.a, k, af);
    tt_->phi_star(a, af, k, phi);
    T.apply(a, phi, k, tb);
    const Mat t2 = y_psi_star(q, ToMat(tt_->iso_full(a), k, std::move(tb)));          // Y_Psi^* T_b Phi_T^*
    const Mat yp = y_phi_star(q, ToMat(tt_->full(a), k, std::move(af)));             // Y_Phi^* alpha
    std::vector<uint32_t> td;
    closed = T.apply_boundary(q, bpos_[q], yp.a, k, td);
    const Mat t1 = psi_d_star(q, ToMat(isob_rank(q), k, std::move(td)));             // Psi_d^* T_d Y_Phi^*
    // eta_T = - Psi_d^* T_d Y_Phi^* - Y_Psi^* T_b Phi_T^* (prop:treepair(4)(a)), assembled from the primitives and NEVER from a planted
    // fault: certify_transport compares it with the off-diagonal block of the operator apply_hecke applies (the D5c audit's m1)
    Mat out(t1.r, k);
    for (size_t i = 0; i < out.a.size(); ++i) out.a[i] = Z_.sub(Z_.sub(0, t1.a[i]), t2.a[i]);
    return out;
}


bool TreePairCochains::certify_transport(const IsoCochainOp& T, int j, const Mat& X, uint32_t prec, bool entrywise,
                                         TreePairCertCounts& cnt, bool force_generators, uint32_t random_vectors) const {
    ENGINE_CHECK_MSG(j >= 0, "certify_transport: cone degree >= 0");
    const uint32_t a = static_cast<uint32_t>(j + 1), q = static_cast<uint32_t>(j);
    const uint32_t ca = X_.c[a], cbq = X_.cb[q], k = X.c;
    const ZMod Zp(Z_.ell, std::max<uint32_t>(1, std::min(prec, Z_.k)));
    const bool full = !force_generators && (entrywise || D_ <= kEtaEntrywiseMaxD);
    const bool eta_reach = a + 1 <= std::min(T_, hecke3::kTreeComparisonTop) && a + 1 <= T.q_top();
    auto ident = [](uint32_t w) {
        Mat I(w, w);
        for (uint32_t i = 0; i < w; ++i) I.at(i, i) = 1;
        return I;
    };
    bool rc = false;
    // (i) the blocks of the APPLIED operator (the D5c audit's m1): apply_hecke(alpha, 0) = (T'_T alpha, eta_T alpha) and
    // apply_hecke(0, c) = (0, T'_d c), each side assembled separately (bulk_hecke, eta, bdry_hecke), compared entrywise in Z/ell^K;
    // (iii) the eta identity d_B eta^j + eta^{j+1} d_T = T'_d a^* - a^* T'_T on the same alphas, reusing the eta_T alpha and T'_T alpha
    // just compared (needs the comparisons to degree j + 2; above them -- the top cone degree at q_max = 5 -- the degree is COUNTED as
    // skipped, never silently: the D5c audit's m2).  `how` names the columns in a failure.  false = T~ not closed on Isob.
    auto check_on = [&](const Mat& A, const Mat& C, const std::string& how) -> bool {
        bool c0 = true, c1 = true;
        const Mat bulkA = bulk_hecke(T, a, A);
        const Mat etaA = eta(T, j, A, c0);
        {
            const Mat R = apply_hecke(T, j, Stack(A, Mat(cbq, A.c)), rc);
            if (rc || !c0) return false;
            if (!EqualMod(Z_, Rows(R, 0, ca), bulkA))
                Fail("the applied T^c_T(alpha, 0) has first component != T'_T alpha at cone degree " + std::to_string(j) + " (prop:treepair(4)(a)) " + how);
            if (!EqualMod(Z_, Rows(R, ca, cbq), etaA))
                Fail("the applied T^c_T(alpha, 0) has off-diagonal block != eta_T alpha at cone degree " + std::to_string(j) +
                     " (prop:treepair(4)(a)) " + how);
            cnt.block_entries += uint64_t(R.r) * R.c;
        }
        {
            const Mat R = apply_hecke(T, j, Stack(Mat(ca, C.c), C), rc);
            const Mat Bd = bdry_hecke(T, q, C, c1);
            if (rc || !c1) return false;
            for (uint32_t i = 0; i < ca; ++i)
                for (uint32_t c = 0; c < C.c; ++c) {
                    ++cnt.strict_entries;
                    if (Z_.red(R.at(i, c))) Fail("the strict block of T^c_T fails (T^c(0, c) has a non-zero first component), prop:treepair(4)(a) " + how);
                }
            if (!EqualMod(Z_, Rows(R, ca, cbq), Bd))
                Fail("the applied T^c_T(0, c) has second component != T'_d c at cone degree " + std::to_string(j) + " (prop:treepair(4)(a)) " + how);
            cnt.block_entries += uint64_t(R.r) * R.c;
        }
        if (eta_reach && A.c) {
            const auto t0 = clk::now();
            bool c2 = true, c3 = true;
            const Mat lhs = Add(Z_, pair::MatMulZ(Z_, X_.db[q], etaA), eta(T, j + 1, pair::MatMulZ(Z_, X_.d[a], A), c2));
            const Mat rhs = Sub(Z_, bdry_hecke(T, a, pair::MatMulZ(Z_, X_.r[a], A), c3), pair::MatMulZ(Z_, X_.r[a], bulkA));
            if (!c2 || !c3) return false;
            if (!EqualMod(Z_, lhs, rhs))
                Fail("the eta identity d eta + eta d = T'_d a^* - a^* T'_T fails at cone degree " + std::to_string(j) + " (prop:treepair(4)(a)) " + how);
            cnt.eta_entries += uint64_t(lhs.r) * lhs.c;
            cnt.eta_seconds += since(t0);
        }
        return true;
    };
    if (full) {
        if (!check_on(ident(ca), ident(cbq), "[entrywise]")) return false;
    } else {
        if (!check_on(Rows(X, 0, ca), Rows(X, ca, cbq), "[on the generators]")) return false;
        // the generators' alpha-parts are d_T-cocycles and the identity's d_B eta term vanishes on d_B-cocycle-valued errors: an error
        // shared by the applied operator and eta_T with such values is invisible there (the C-D8b referee's Arm (G)); the random vectors
        // see all of C^{j+1}_T and B^j (error bound at kFreivaldsVectors)
        if (random_vectors) {
            const uint64_t seed = (uint64_t(Z_.ell) << 56) ^ (uint64_t(N_) << 32) ^ (uint64_t(n_) << 12) ^ uint64_t(j + 1);
            if (!check_on(RandomMat(Z_, ca, random_vectors, seed), RandomMat(Z_, cbq, random_vectors, seed ^ 0x5bd1e995ull),
                          "[on " + std::to_string(random_vectors) + " random vectors]"))
                return false;
            cnt.random_columns += random_vectors;
        }
    }
    if (eta_reach) {
        cnt.eta_entrywise = cnt.eta_entrywise || full;
        ++cnt.eta_degrees;
    } else {
        ++cnt.eta_skipped;
        cnt.eta_skipped_top = std::max(cnt.eta_skipped_top, j);
    }
    // (ii) D T^c g = 0 for the columns g of X (cocycles of Cone(a^*)^j), mod ell^prec; D = (-d, d_B + a^*)
    if (a + 1 <= T_) {
        const Mat R = apply_hecke(T, j, X, rc);
        if (rc) return false;
        const Mat al = Rows(R, 0, ca), cc = Rows(R, ca, cbq);
        const Mat first = pair::MatMulZ(Z_, X_.d[a], al);
        const Mat second = Add(Z_, pair::MatMulZ(Z_, X_.db[q], cc), pair::MatMulZ(Z_, X_.r[a], al));
        if (!IsZero(Zp, first) || !IsZero(Zp, second))
            Fail("T^c_T maps a cocycle of Cone(a^*)^" + std::to_string(j) + " to a non-cocycle (mod " + std::to_string(Z_.ell) + "^" +
                 std::to_string(Zp.k) + "): T^c_T is not a cochain map, prop:treepair(4)(a), (b)");
        cnt.cocycle_columns += k;
    }
    return true;
}

// ---------------------------------------------------------------------------------------------------------------------------------
// the read
// ---------------------------------------------------------------------------------------------------------------------------------
Mat InducedFromImages(const pair::Pres& X, const pair::Pres& Y, const Mat& images, ZMod& Zout) {
    ENGINE_CHECK_MSG(images.r == Y.c && images.c == X.g(), "InducedFromImages: shapes");
    const ZMod Z = X.Z.k <= Y.Z.k ? X.Z : Y.Z;
    Zout = Z;
    Mat Yc = Y.coord;
    for (auto& x : Yc.a) x = Z.red(x);
    Mat Im = images;
    for (auto& x : Im.a) x = Z.red(x);
    Mat F = pair::MatMulZ(Z, Yc, Im);
    for (uint32_t i = 0; i < Y.tors.size(); ++i) {
        const ZMod Zi(Z.ell, Y.tors[i]);
        for (uint32_t j = 0; j < F.c; ++j) F.at(i, j) = Zi.red(F.at(i, j));
    }
    return F;
}

namespace {
pair::PairRefusal FromD3(hecke3::D3Refusal c, pair::PairRefusal lift) {
    switch (c) {
        case hecke3::D3Refusal::NotPrime: return pair::PairRefusal::NotPrime;
        case hecke3::D3Refusal::PDividesN: return pair::PairRefusal::PDividesN;
        case hecke3::D3Refusal::DegreeRange: return pair::PairRefusal::DegreeRange;
        case hecke3::D3Refusal::Lift: return lift;
        case hecke3::D3Refusal::IntegerCap: return pair::PairRefusal::IntegerCap;
        case hecke3::D3Refusal::DomainCap: return pair::PairRefusal::DomainCap;
        default: break;
    }
    // no other D3 code arises in IsoHeckeN's construction
    throw hecke::HeckeInternalError(("ReadTreePairHecke: an unexpected refusal code of IsoHeckeN: " + std::string(hecke3::D3RefusalName(c))).c_str());
}
uint32_t MaxTors(const pair::Pres& P) {
    uint32_t m = 0;
    for (uint32_t a : P.tors) m = std::max(m, a);
    return m;
}
}  // namespace

void DenseIsoOp::apply(uint32_t q, const std::vector<uint32_t>& X, uint32_t k, std::vector<uint32_t>& Y) const {
    const Mat& A = Tb_.at(q);
    Mat In;
    In.r = A.c;
    In.c = k;
    In.a = X;
    ENGINE_CHECK_MSG(X.size() == size_t(A.c) * k, "DenseIsoOp::apply: shapes");
    Y = pair::MatMulZ(Z_, A, In).a;
}
bool DenseIsoOp::apply_boundary(uint32_t q, const std::vector<int32_t>& bpos, const std::vector<uint32_t>& X, uint32_t k,
                                std::vector<uint32_t>& Y) const {
    const Mat& A = Tb_.at(q);
    const uint32_t W = static_cast<uint32_t>(bpos.size());
    ENGINE_CHECK_MSG(A.r == W * D_ && A.c == W * D_, "DenseIsoOp::apply_boundary: the layout");
    std::vector<uint32_t> rows, cols;
    bool closed = true;
    for (uint32_t c = 0; c < W; ++c)
        if (bpos[c] >= 0)
            for (uint32_t a = 0; a < D_; ++a) rows.push_back(c * D_ + a);
    for (uint32_t r : rows)
        for (uint32_t c = 0; c < W; ++c)
            if (bpos[c] < 0)
                for (uint32_t a = 0; a < D_; ++a)
                    if (Z_.red(A.at(r, c * D_ + a))) closed = false;
    Mat B(static_cast<uint32_t>(rows.size()), static_cast<uint32_t>(rows.size()));
    for (uint32_t i = 0; i < rows.size(); ++i)
        for (uint32_t j = 0; j < rows.size(); ++j) B.at(i, j) = A.at(rows[i], rows[j]);
    Mat In;
    In.r = B.c;
    In.c = k;
    In.a = X;
    ENGINE_CHECK_MSG(X.size() == size_t(B.c) * k, "DenseIsoOp::apply_boundary: shapes");
    Y = pair::MatMulZ(Z_, B, In).a;
    return closed;
}

void TransportOnPresentations(const TreePairCochains& C, const pair::PairLocal& L, uint32_t qm, const IsoCochainOp& T, bool eta_entrywise,
                              TreePairHeckeOp& op, bool force_generators, uint32_t random_vectors) {
    const auto t0 = clk::now();
    auto refuse = [&](pair::PairRefusal code, const std::string& w) {
        op.refused = true;
        op.code = code;
        op.refusal = w;
    };
    ENGINE_CHECK_MSG(qm < L.H.size() && qm < L.Hd.size() && qm < L.Hc.size(), "TransportOnPresentations: the presentations stop below q_max");
    try {
        bool closed_all = true;
        op.on_H.assign(qm + 1, Mat());
        op.on_Hd.assign(qm + 1, Mat());
        op.on_Hc.assign(qm + 1, Mat());
        op.Z_H.assign(qm + 1, ZMod());
        op.Z_Hd.assign(qm + 1, ZMod());
        op.Z_Hc.assign(qm + 1, ZMod());
        for (uint32_t q = 0; q <= qm && closed_all; ++q) {
            const pair::Pres& H = L.H[q];
            if (H.g()) op.on_H[q] = InducedFromImages(H, H, C.bulk_hecke(T, q, Transpose(H.gensT)), op.Z_H[q]);
            const pair::Pres& Hd = L.Hd[q];
            if (Hd.g()) {
                bool cl = true;
                op.on_Hd[q] = InducedFromImages(Hd, Hd, C.bdry_hecke(T, q, Transpose(Hd.gensT), cl), op.Z_Hd[q]);
                closed_all = closed_all && cl;
            }
            if (q >= 1 && closed_all) {
                const pair::Pres& Hc = L.Hc[q];
                const int j = static_cast<int>(q) - 1;
                if (Hc.g()) {
                    const Mat G = Transpose(Hc.gensT);
                    bool rc = false;
                    const Mat img = C.apply_hecke(T, j, G, rc);
                    closed_all = closed_all && !rc;
                    if (!closed_all) break;
                    if (!C.certify_transport(T, j, G, Hc.Z.k > MaxTors(Hc) ? Hc.Z.k - MaxTors(Hc) : 1u, eta_entrywise, op.cert, force_generators, random_vectors)) {
                        closed_all = false;
                        break;
                    }
                    op.on_Hc[q] = InducedFromImages(Hc, Hc, img, op.Z_Hc[q]);
                } else {
                    // H^q_c = 0 here: no generator, but the eta identity of this cone degree is still checked (entrywise when D is
                    // below kEtaEntrywiseMaxD; on no column above)
                    if (!C.certify_transport(T, j, Mat(Hc.c, 0), 1u, eta_entrywise, op.cert, force_generators, random_vectors)) {
                        closed_all = false;
                        break;
                    }
                }
            }
        }
        if (!closed_all) refuse(pair::PairRefusal::IsobClosure, "the operator does not preserve Isob on this layout (tau_Iso(Isob) not in Isob)");
    } catch (const hecke::HeckeInternalError&) {
        throw;
    } catch (const std::exception& e) {
        const pair::PairRefusal c = pair::ClassifyPairException(e);
        if (c == pair::PairRefusal::None) throw;
        refuse(c, std::string("the transport stopped: ") + e.what());
    }
    op.seconds += since(t0);
}

TreePairHeckeRead ReadTreePairHecke(uint32_t ell, uint32_t N, uint32_t n, const std::vector<uint32_t>& ps, const TreePairHeckeOptions& opt) {
    const auto t0 = clk::now();
    TreePairHeckeRead R;
    R.ell = ell;
    R.N = N;
    R.n = n;
    R.q_max = opt.q_max;
    R.D = PsiIndex(N) * (n + 1);
    auto refuse_all = [&](pair::PairRefusal code, const std::string& w) {
        R.refused = true;
        R.refusal = "REFUSED [" + std::string(pair::PairRefusalName(code)) + "] the tree pair at (ell, N, n) = (" + std::to_string(ell) + ", " +
                    std::to_string(N) + ", " + std::to_string(n) + "): " + w;
        for (uint32_t p : ps) {
            TreePairHeckeOp op;
            op.p = p;
            op.refused = true;
            op.code = code;
            op.refusal = R.refusal;
            R.ops.push_back(op);
        }
        R.seconds = since(t0);
        return R;
    };
    if (opt.q_max != 4 && opt.q_max != 5) return refuse_all(pair::PairRefusal::DegreeRange, "q_max = " + std::to_string(opt.q_max) + " not in {4, 5} (prop:treepair(6))");
    const uint32_t T = opt.q_max + 1;
    R.rss_pred = PredictTreePairRss(ell, R.D, T);
    if (R.rss_pred > opt.rss_cap_bytes / 10 * 9)
        return refuse_all(pair::PairRefusal::RssCeiling, "rule 19: the tree pair predicts " + std::to_string(R.rss_pred >> 20) + " MiB above 0.9 x the ceiling (refused before building)");
    TreePairCochains C(ell, N, n, T);
    const pair::PairBetti bt = pair::PairBettiOf(N, n);
    R.rep.ell = ell;
    R.rep.min_prec = 1000;
    if (!pair::RunPairLes(C.tree(), bt, opt.q_max, R.L, R.rep)) {
        R.refused = true;
        R.refusal = "REFUSED the tree pair's sequence at (ell, N, n) = (" + std::to_string(ell) + ", " + std::to_string(N) + ", " + std::to_string(n) + "): " + R.rep.why;
        R.seconds = since(t0);
        return R;
    }
    const uint32_t qm = opt.q_max;
    // rule 19 for the transport (the D5c audit's N2): on the built layout, before any operator is formed -- the checked columns are every
    // basis vector of the widest C^{j+1}_T and B^j when the identities run entrywise, else the presentations' generators
    {
        uint32_t cols = std::max<uint32_t>(1, opt.random_vectors);
        const bool full = !opt.force_generators && (opt.eta_entrywise || R.D <= kEtaEntrywiseMaxD);
        for (uint32_t q = 1; q <= qm; ++q) {
            cols = std::max(cols, R.L.Hc[q].g());
            if (full) cols = std::max(cols, std::max(C.tree().c[q], C.tree().cb[q - 1]));
        }
        R.rss_pred_transport = C.transport_bytes(qm, cols);
        R.rss_pred += R.rss_pred_transport;
        if (R.rss_pred > opt.rss_cap_bytes / 10 * 9)
            return refuse_all(pair::PairRefusal::RssCeiling, "rule 19: the tree pair with its transport predicts " + std::to_string(R.rss_pred >> 20) +
                                                                 " MiB above 0.9 x the ceiling (the groups built, no operator formed)");
    }
    for (uint32_t p : ps) {
        const auto tp = clk::now();
        TreePairHeckeOp op;
        op.p = p;
        auto refuse = [&](pair::PairRefusal code, const std::string& w) {
            op.refused = true;
            op.code = code;
            op.refusal = "REFUSED [" + std::string(pair::PairRefusalName(code)) + "] T_" + std::to_string(p) + " on the tree pair at (ell, N, n) = (" +
                         std::to_string(ell) + ", " + std::to_string(N) + ", " + std::to_string(n) + "): " + w;
        };
        // the operator on Iso to cochain degree T (the transport reads it to degree T; the lift is built to degree 6)
        std::unique_ptr<hecke3::IsoHeckeN> Tp;
        try {
            Tp = std::make_unique<hecke3::IsoHeckeN>(N, n, p, C.ctx(), T);
        } catch (const std::exception& e) {
            const pair::PairRefusal c = pair::ClassifyPairException(e);
            if (c == pair::PairRefusal::None) throw;
            refuse(c, e.what());
        }
        if (!op.refused && Tp->refused()) refuse(FromD3(Tp->code(), Tp->lift_code()), Tp->why());
        if (!op.refused) {
            const IsoHeckeOp opT(*Tp);
            TransportOnPresentations(C, R.L, qm, opT, opt.eta_entrywise, op, opt.force_generators, opt.random_vectors);
            if (op.refused) op.refusal = "REFUSED [" + std::string(pair::PairRefusalName(op.code)) + "] T_" + std::to_string(p) + " on the tree pair at (ell, N, n) = (" +
                                         std::to_string(ell) + ", " + std::to_string(N) + ", " + std::to_string(n) + "): " + op.refusal;
        }
        op.seconds = since(tp);
        R.ops.push_back(std::move(op));
    }
    R.seconds = since(t0);
    return R;
}

}  // namespace treepair
}  // namespace integral_descent
