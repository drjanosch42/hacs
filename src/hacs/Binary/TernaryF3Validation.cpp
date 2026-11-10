// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

#include "TernaryF3Validation.h"

#include <algorithm>
#include <memory>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "../../ModArithmetic/Binary/BitMatrix.h"
#include "../../ModArithmetic/Binary/TritMatrix.h"
#include "../../ModArithmetic/DenseBuf.h"
#include "../../ModArithmetic/DenseElim.h"
#include "../../ModArithmetic/DenseView.h"
#include "../../ModArithmetic/FlatMatrix.h"
#include "../../ModArithmetic/Kernels.h"
#include "../../ModArithmetic/ModCtx.h"
#include "../../ModArithmetic/RefLinAlg.h"
#include "../EquivariantCohomologyOnK.h"
#include "../LocalSystemSL2Z.h"
#include "../LocalSystemSimplicialComplex.h"
#include "../SimplicialCohomology.h"
#include "../Orbifold/L23/AmalgamReader.h"

// See TernaryF3Validation.h.  The structure follows BinaryF2Validation (slot-22 BIN-1).

namespace {

using engine::DenseBuf;
using engine::cview;
using engine::view;
namespace bin = engine::binary;
namespace elim = engine::elim;

uint32_t g_fail = 0;
uint64_t g_checks = 0;
void check(bool ok, const std::string& what) {
    ++g_checks;
    if (!ok) {
        ++g_fail;
        if (g_fail <= 40) std::cout << "  [FAIL] " << what << std::endl;
    }
}
template<class T> std::string str(T v) { return std::to_string(v); }

uint64_t splitmix(uint64_t& s) {
    uint64_t z = (s += 0x9e3779b97f4a7c15ULL);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
}
double now_s() {
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

const engine::ModCtx<uint32_t>& ctx3() { static const engine::ModCtx<uint32_t> c(3u); return c; }

// density in 1/256 units: an entry is non-zero with probability dens / 256, then 1 or 2 equally.
DenseBuf<uint32_t> random_dense(uint32_t m, uint32_t n, uint32_t dens, uint64_t& s) {
    DenseBuf<uint32_t> A(m, n);
    for (uint32_t i = 0; i < m; ++i)
        for (uint32_t j = 0; j < n; ++j) {
            const uint64_t r = splitmix(s);
            A.at(i, j) = (r & 255u) < dens ? 1u + static_cast<uint32_t>((r >> 8) & 1u) : 0u;
        }
    return A;
}
DenseBuf<uint32_t> low_rank(uint32_t m, uint32_t n, uint32_t r, uint64_t& s) {
    const DenseBuf<uint32_t> L = random_dense(m, r, 170, s), R = random_dense(r, n, 170, s);
    DenseBuf<uint32_t> C(m, n);
    engine::kernels::gemm<uint32_t>(ctx3(), cview(L), cview(R), view(C), engine::kernels::GemmOp::Set);
    return C;
}

DenseBuf<uint32_t> unpack(const bin::TritMatrix& T) {
    DenseBuf<uint32_t> D(T.rows(), T.cols());
    T.to_dense<uint32_t>(view(D));
    return D;
}

std::vector<bin::RowPath> available_paths() {
    std::vector<bin::RowPath> v;
    for (bin::RowPath p : {bin::RowPath::Portable, bin::RowPath::Avx2, bin::RowPath::Avx512})
        if (bin::row_path_available(p)) v.push_back(p);
    return v;
}
struct PathGuard {   // restores Auto on every exit
    ~PathGuard() { bin::set_row_path(bin::RowPath::Auto); }
};

// ---------------------------------------------------------------------------
// Part 0: the arithmetic, exhaustively
// ---------------------------------------------------------------------------
void arithmetic_part() {
    PathGuard g;
    for (bin::RowPath p : available_paths()) {
        bin::set_row_path(p);
        const std::string pt = std::string("[") + bin::row_path_name(p) + "/" + bin::ternary_row_body_name() + "]";
        // planes: rows x (1 x 600) built so that column j holds the pair (j % 3, (j / 3) % 3)
        const uint32_t n = 600;
        bin::TritMatrix X(2, n), Y(1, n);
        for (uint32_t j = 0; j < n; ++j) { X.set(0, j, j % 3); X.set(1, j, (j / 3) % 3); Y.set(0, j, (j / 3) % 3); }
        // X row 1 += ... through multiply: [1 1] * [x; y] = x + y, [1 2] * [x; y] = x - y
        bin::TritMatrix Ia(1, 2), Is(1, 2);
        Ia.set(0, 0, 1); Ia.set(0, 1, 1);
        Is.set(0, 0, 1); Is.set(0, 1, 2);
        const bin::TritMatrix S = bin::multiply(Ia, X), D = bin::multiply(Is, X);
        bool ok_s = true, ok_d = true;
        for (uint32_t j = 0; j < n; ++j) {
            const uint32_t x = j % 3, y = (j / 3) % 3;
            ok_s = ok_s && S.get(0, j) == (x + y) % 3;
            ok_d = ok_d && D.get(0, j) == (x + 3 - y) % 3;
        }
        check(ok_s, pt + ": bit-sliced x + y on all 9 pairs (600 columns)");
        check(ok_d, pt + ": bit-sliced x - y on all 9 pairs (600 columns)");
        bin::TritMatrix Nx = X;
        Nx.negate_row(0);
        bool ok_n = true;
        for (uint32_t j = 0; j < n; ++j) ok_n = ok_n && Nx.get(0, j) == (3 - j % 3) % 3;
        check(ok_n, pt + ": plane swap = negation");
        // bytes: every length 0..200 (all tails), all 9 pairs
        bool ok_b = true;
        for (uint32_t len = 0; len <= 200 && ok_b; ++len) {
            std::vector<uint8_t> a(len), b(len), c, e;
            for (uint32_t i = 0; i < len; ++i) { a[i] = static_cast<uint8_t>(i % 3); b[i] = static_cast<uint8_t>((i / 3 + len) % 3); }
            c = a; bin::f3_add_u8(c.data(), b.data(), len);
            for (uint32_t i = 0; i < len; ++i) ok_b = ok_b && c[i] == (a[i] + b[i]) % 3;
            c = a; bin::f3_sub_u8(c.data(), b.data(), len);
            for (uint32_t i = 0; i < len; ++i) ok_b = ok_b && c[i] == (a[i] + 3 - b[i]) % 3;
            c = a; bin::f3_neg_u8(c.data(), len);
            for (uint32_t i = 0; i < len; ++i) ok_b = ok_b && c[i] == (3 - a[i]) % 3;
        }
        check(ok_b, pt + ": byte add / sub / neg on every length 0..200");
    }
}

// ---------------------------------------------------------------------------
// Part 1: one matrix against the engine at p = 3 (ENGINE_API.md Sec. 8.3's contract)
// ---------------------------------------------------------------------------
void compare_one(const std::string& tag, const DenseBuf<uint32_t>& A, bool with_oracle, bool with_product, uint64_t& s) {
    const auto& ctx = ctx3();
    DenseBuf<uint32_t> Re = A;
    elim::RrefResult re, crp_e;
    DenseBuf<uint32_t> Ke;
    {
        elim::ScopedTernaryF3 off(false);
        re = elim::rref<uint32_t>(ctx, view(Re));
        Ke = elim::kernel_basis<uint32_t>(ctx, cview(Re), re.pivots);
        DenseBuf<uint32_t> tmp = A;
        crp_e = elim::column_rank_profile<uint32_t>(ctx, view(tmp));
    }
    check(crp_e.pivots == re.pivots, tag + ": engine column_rank_profile == rref pivots");

    const bin::TritMatrix T = bin::TritMatrix::from_dense<uint32_t>(cview(A));
    check(unpack(T).same_entries(A), tag + ": pack / unpack round trip");
    uint64_t fnv_first = 0;
    {
        PathGuard g;
        bool first = true;
        for (bin::RowPath p : available_paths()) {
            bin::set_row_path(p);
            const std::string pt = tag + " [" + bin::row_path_name(p) + "]";
            bin::TritMatrix R = T;
            const bin::TritRref tr = bin::rref(R);
            check(tr.rank == re.rank && tr.pivots == re.pivots, pt + ": rank / pivots == engine");
            if (first) {
                check(unpack(R).same_entries(Re), pt + ": RREF entrywise == engine");
                fnv_first = R.fnv();
                const bin::TritMatrix K = bin::kernel_basis(R, tr.pivots);
                check(K.rows() == Ke.rows && unpack(K).same_entries(Ke), pt + ": kernel basis entrywise == engine");
                bin::TritMatrix Kt(K.cols(), K.rows());
                for (uint32_t i = 0; i < K.rows(); ++i)
                    for (uint32_t j = 0; j < K.cols(); ++j) Kt.set(j, i, K.get(i, j));
                const bin::TritMatrix Z = bin::multiply(T, Kt);
                bool zero = true;
                for (uint32_t i = 0; i < Z.rows() && zero; ++i)
                    for (uint32_t j = 0; j < Z.cols() && zero; ++j) zero = Z.get(i, j) == 0;
                check(zero, pt + ": A * kernel^T == 0");
                first = false;
            } else {
                check(R.fnv() == fnv_first, pt + ": RREF == the first path's");
            }
            bin::TritMatrix E = T;
            const bin::TritRref te = bin::echelon(E);
            check(te.rank == re.rank && te.pivots == re.pivots, pt + ": echelon pivots == engine");
            check(bin::rank(T) == re.rank, pt + ": rank() == engine");
        }
    }
    if (with_product) {   // strided (transposed-view) pack / unpack
        const bin::TritMatrix Tt = bin::TritMatrix::from_dense<uint32_t>(cview(A).transposed_view());
        bool ok = Tt.rows() == A.cols && Tt.cols() == A.rows;
        for (uint32_t i = 0; ok && i < A.rows; ++i)
            for (uint32_t j = 0; ok && j < A.cols; ++j) ok = Tt.get(j, i) == A.at(i, j);
        DenseBuf<uint32_t> back(A.rows, A.cols);
        Tt.to_dense<uint32_t>(view(back).transposed_view());
        check(ok && back.same_entries(A), tag + ": strided (transposed-view) pack / unpack round trip");
        std::vector<uint8_t> u8(static_cast<size_t>(A.rows) * A.cols);
        for (uint32_t i = 0; i < A.rows; ++i)
            for (uint32_t j = 0; j < A.cols; ++j) u8[static_cast<size_t>(i) * A.cols + j] = static_cast<uint8_t>(A.at(i, j));
        const bin::TritMatrix Tu = bin::TritMatrix::from_u8(u8.data(), A.rows, A.cols, A.cols);
        std::vector<uint8_t> u8b(u8.size(), 7);
        Tu.to_u8(u8b.data(), A.cols);
        check(Tu == T && u8b == u8, tag + ": from_u8 / to_u8 round trip == from_dense");
    }
    if (with_oracle && static_cast<uint64_t>(A.rows) * A.cols <= 20000) {
        bin::TritMatrix R = T;
        const bin::TritRref tn = bin::rref_naive(R);
        check(tn.pivots == re.pivots && R.fnv() == fnv_first, tag + ": rref_naive == Gauss--Jordan");
        DenseBuf<uint32_t> Rr = A;
        const std::vector<uint32_t> pv = engine::ref::rref<uint32_t>(3u, view(Rr));
        check(pv == re.pivots && Rr.same_entries(Re), tag + ": RREF / pivots == RefLinAlg");
    }
    {   // the dispatched route of engine::elim
        elim::ScopedTernaryF3 on(true);
        const elim::TernaryF3Totals t0 = elim::ternary_f3_totals();
        DenseBuf<uint32_t> Rd = A;
        const elim::RrefResult rd = elim::rref<uint32_t>(ctx, view(Rd));
        DenseBuf<uint32_t> tmp = A;
        const elim::RrefResult cd = elim::column_rank_profile<uint32_t>(ctx, view(tmp));
        const elim::TernaryF3Totals t1 = elim::ternary_f3_totals();
        const bool nonempty = A.rows > 0 && A.cols > 0;
        check(rd.rank == re.rank && rd.pivots == re.pivots && Rd.same_entries(Re), tag + ": routed elim::rref == engine");
        check(cd.pivots == re.pivots && cd.rank == re.rank, tag + ": routed column_rank_profile == engine");
        check(!nonempty || (t1.rrefs == t0.rrefs + 1 && t1.rank_profiles == t0.rank_profiles + 1),
              tag + ": route counters advanced (the route really ran)");
    }
    if (with_product) {
        const uint32_t k = 1 + static_cast<uint32_t>(splitmix(s) % 150);
        const DenseBuf<uint32_t> Y = random_dense(A.cols, k, 170, s);
        DenseBuf<uint32_t> Ce(A.rows, k);
        engine::kernels::gemm<uint32_t>(ctx, cview(A), cview(Y), view(Ce), engine::kernels::GemmOp::Set);
        const bin::TritMatrix Yt = bin::TritMatrix::from_dense<uint32_t>(cview(Y));
        PathGuard g;
        for (bin::RowPath p : available_paths()) {
            bin::set_row_path(p);
            check(unpack(bin::multiply(T, Yt)).same_entries(Ce), tag + " [" + bin::row_path_name(p) + "]: product == engine gemm");
        }
        if (static_cast<uint64_t>(A.rows) * A.cols * k <= 4000000)
            check(unpack(bin::multiply_naive(T, Yt)).same_entries(Ce), tag + ": naive product == engine gemm");
    }
}

// The uint64_t instantiation of the route, and the other entry points of engine::elim that sit on the routed RREF.
void compare_u64(const std::string& tag, const DenseBuf<uint32_t>& A32) {
    const engine::ModCtx<uint64_t> ctx(3u);
    DenseBuf<uint64_t> A(A32.rows, A32.cols);
    for (uint32_t i = 0; i < A.rows; ++i)
        for (uint32_t j = 0; j < A.cols; ++j) A.at(i, j) = A32.at(i, j);
    DenseBuf<uint64_t> Re = A, Rd = A;
    elim::RrefResult re, rd;
    { elim::ScopedTernaryF3 off(false); re = elim::rref<uint64_t>(ctx, view(Re)); }
    { elim::ScopedTernaryF3 on(true);   rd = elim::rref<uint64_t>(ctx, view(Rd)); }
    check(re.pivots == rd.pivots && re.rank == rd.rank && Re.same_entries(Rd), tag + ": uint64_t routed rref == engine");
    // rref_with_transform (U = trailing columns of RREF([A | I]))
    const engine::ModCtx<uint32_t>& c32 = ctx3();
    DenseBuf<uint32_t> R0(A32.rows, A32.cols), U0(A32.rows, A32.rows), R1(A32.rows, A32.cols), U1(A32.rows, A32.rows);
    elim::RrefResult w0, w1;
    { elim::ScopedTernaryF3 off(false); w0 = elim::rref_with_transform<uint32_t>(c32, cview(A32), view(R0), view(U0)); }
    { elim::ScopedTernaryF3 on(true);   w1 = elim::rref_with_transform<uint32_t>(c32, cview(A32), view(R1), view(U1)); }
    check(w0.pivots == w1.pivots && R0.same_entries(R1) && U0.same_entries(U1), tag + ": routed rref_with_transform == engine");
    {   // (TER-1 audit F3) solve_multi_rhs on Y = A X (in the span) and complement_basis([A | I]) with its projector
        uint64_t s = 0x50175ULL + A32.rows * 131u + A32.cols;
        const uint32_t k = 1 + static_cast<uint32_t>(splitmix(s) % 8);
        const DenseBuf<uint32_t> X = random_dense(A32.cols, k, 170, s);
        DenseBuf<uint32_t> Y(A32.rows, k);
        engine::kernels::gemm<uint32_t>(c32, cview(A32), cview(X), view(Y), engine::kernels::GemmOp::Set);
        std::vector<std::vector<uint32_t>> y0, y1;
        std::string e0, e1;
        { elim::ScopedTernaryF3 off(false); try { y0 = elim::solve_multi_rhs<uint32_t>(c32, cview(A32), cview(Y)); } catch (const std::exception& e) { e0 = e.what(); } }
        { elim::ScopedTernaryF3 on(true);   try { y1 = elim::solve_multi_rhs<uint32_t>(c32, cview(A32), cview(Y)); } catch (const std::exception& e) { e1 = e.what(); } }
        check(e0.empty() && e1.empty() && y0 == y1, tag + ": routed solve_multi_rhs == engine");
        DenseBuf<uint32_t> G(A32.rows, A32.rows);
        for (uint32_t i = 0; i < A32.rows; ++i) G.at(i, i) = 1;
        DenseBuf<uint32_t> P0, P1;
        std::vector<uint32_t> c0, c1;
        { elim::ScopedTernaryF3 off(false); c0 = elim::complement_basis<uint32_t>(c32, cview(A32), cview(G), &P0); }
        { elim::ScopedTernaryF3 on(true);   c1 = elim::complement_basis<uint32_t>(c32, cview(A32), cview(G), &P1); }
        check(c0 == c1 && P0.rows == P1.rows && P0.cols == P1.cols && P0.same_entries(P1), tag + ": routed complement_basis (+ projector) == engine");
    }
    if (A32.rows == A32.cols) {
        DenseBuf<uint32_t> I0(A32.rows, A32.cols), I1(A32.rows, A32.cols);
        bool ok0, ok1;
        { elim::ScopedTernaryF3 off(false); ok0 = elim::inverse<uint32_t>(c32, cview(A32), view(I0)); }
        { elim::ScopedTernaryF3 on(true);   ok1 = elim::inverse<uint32_t>(c32, cview(A32), view(I1)); }
        check(ok0 == ok1 && (!ok0 || I0.same_entries(I1)), tag + ": routed inverse == engine");
    }
}

uint32_t random_part() {
    uint64_t s = 0x7E3F3ULL;
    const uint32_t dens_list[] = {170, 128, 26, 5, 240, 64};
    const uint32_t edge_n[] = {1, 63, 64, 65, 127, 128, 129, 511, 512, 513};
    uint32_t count = 0;
    for (uint32_t t = 0; t < 400; ++t) {
        uint32_t m, n;
        if (t < 20) {
            m = (t & 1) ? 1 + static_cast<uint32_t>(splitmix(s) % 80) : edge_n[(t / 2) % 10];
            n = (t & 1) ? edge_n[(t / 2) % 10] : 1 + static_cast<uint32_t>(splitmix(s) % 80);
        } else if (t % 50 == 7) {
            m = 300 + static_cast<uint32_t>(splitmix(s) % 700);
            n = 300 + static_cast<uint32_t>(splitmix(s) % 700);
        } else if (t % 9 == 4) {
            n = 1 + static_cast<uint32_t>(splitmix(s) % 150);
            m = n;
        } else {
            m = 1 + static_cast<uint32_t>(splitmix(s) % 200);
            n = 1 + static_cast<uint32_t>(splitmix(s) % 200);
        }
        DenseBuf<uint32_t> A;
        std::string kind;
        if (t == 20) { A = DenseBuf<uint32_t>(m, n); kind = "zero"; }
        else if (t == 21) { A = DenseBuf<uint32_t>(n, n); for (uint32_t i = 0; i < n; ++i) A.at(i, i) = 2; kind = "2*identity"; }
        else if (t % 4 == 3) {
            const uint32_t r = static_cast<uint32_t>(splitmix(s) % (std::min(m, n) + 1));
            A = low_rank(m, n, std::max<uint32_t>(r, 1), s);
            kind = "lowrank" + str(r);
        } else {
            const uint32_t d = dens_list[t % 6];
            A = random_dense(m, n, d, s);
            kind = "dens" + str(d);
        }
        const std::string tag = "random#" + str(t) + " " + str(m) + "x" + str(n) + " " + kind;
        compare_one(tag, A, t % 5 == 0, t % 10 == 0, s);
        if (t % 9 == 4 || t % 25 == 0) compare_u64(tag, A);
        ++count;
    }
    // (TER-1 audit R5) the boundary x boundary grid, zero dimensions included
    const uint32_t bnd[] = {0, 1, 63, 64, 65, 511, 512, 513};
    for (uint32_t m : bnd)
        for (uint32_t n : bnd) {
            const DenseBuf<uint32_t> A = (m == 0 || n == 0) ? DenseBuf<uint32_t>(m, n) : (m * 7 + n) % 3 == 0 ? low_rank(m, n, std::max<uint32_t>(1, std::min(m, n) / 3), s)
                                                              : random_dense(m, n, (m + n) % 2 ? 170 : 13, s);
            compare_one("boundary " + str(m) + "x" + str(n), A, false, m > 0 && n > 0 && m * n <= 70000, s);
            ++count;
        }
    return count;
}

// ---------------------------------------------------------------------------
// Part 1b: the real A_Cq matrices of the 7-vertex SL_2(Z) model mod 3 (as BinaryF2Validation builds them mod 2)
// ---------------------------------------------------------------------------
bool is_identity(const FlatMatrix& M, uint32_t d) {
    if (M.get_rows() != d || M.get_cols() != d) return false;
    for (uint32_t r = 0; r < d; ++r)
        for (uint32_t c = 0; c < d; ++c)
            if (M.get_entry(r, c) != (r == c ? 1u : 0u)) return false;
    return true;
}
DenseBuf<uint32_t> build_ACq(const LocalSystemSimplicialComplex& lssc, uint32_t q) {
    const SimplicialComplex* K = lssc.get_base_K_complex();
    const uint32_t d = lssc.get_representation_dim();
    const uint32_t ell = lssc.get_prime();
    const uint32_t N_q = K->get_num_q_simplices(q);
    const bool shortcircuit = EquivariantCohomologyOptions::vacuous_relation_shortcircuit_enabled();
    std::vector<const LocalSystemSimplicialComplex::SimplexRelation*> live;
    for (const auto& rel : lssc.get_user_defined_simplex_relations()) {
        if (rel.q_dim != q) continue;
        if (shortcircuit && rel.s_lhs_idx_in_Kq == rel.s_rhs_idx_in_Kq && is_identity(rel.M, d)) continue;
        live.push_back(&rel);
    }
    DenseBuf<uint32_t> A(static_cast<uint32_t>(live.size()) * d, N_q * d);
    uint32_t ri = 0;
    for (const auto* rp : live) {
        const auto& rel = *rp;
        const uint32_t r0 = ri * d;
        for (uint32_t r = 0; r < d; ++r)
            for (uint32_t c = 0; c < d; ++c) {
                const uint32_t mrc = rel.M.get_entry(r, c);
                if (rel.s_lhs_idx_in_Kq == rel.s_rhs_idx_in_Kq) {
                    A.at(r0 + r, rel.s_lhs_idx_in_Kq * d + c) = ModArithmetic::mod_sub(r == c ? 1u : 0u, mrc, ell);
                } else {
                    if (r == c) A.at(r0 + r, rel.s_lhs_idx_in_Kq * d + c) = 1u;
                    A.at(r0 + r, rel.s_rhs_idx_in_Kq * d + c) = ModArithmetic::mod_sub(0u, mrc, ell);
                }
            }
        ++ri;
    }
    return A;
}
std::vector<LocalSystemSimplicialComplex::SimplexRelation> relations(const SimplicialComplex& K, uint32_t n_sym) {
    std::vector<LocalSystemSimplicialComplex::SimplexRelation> rels;
    for (const auto& datum : GenerateSL2ZRelationsWithGroupData(K, n_sym, 3u)) rels.push_back(datum.rel);
    return rels;
}
uint32_t amalgam_part() {
    uint64_t s = 0xA3A16B3ULL;
    const uint32_t n_list[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 16, 17, 26, 27, 28, 63, 64, 81, 100};
    const SimplicialComplex K = CreateSL2ZFundamentalDomainTriangulation();
    uint32_t count = 0;
    for (int sc = 1; sc >= 0; --sc) {
        EquivariantCohomologyOptions::ScopedVacuousShortcircuit guard(sc != 0);
        for (uint32_t n_sym : n_list) {
            const LocalSystemSimplicialComplex lssc(K, relations(K, n_sym), n_sym + 1, 3u);
            for (uint32_t q = 0; q <= 2; ++q) {
                const DenseBuf<uint32_t> A = build_ACq(lssc, q);
                if (A.rows == 0 || A.cols == 0) continue;
                const std::string tag = "A_C" + str(q) + " n_sym=" + str(n_sym) + (sc ? " sc=on " : " sc=off ") +
                                        str(A.rows) + "x" + str(A.cols);
                compare_one(tag, A, true, count % 7 == 0, s);
                if (count % 10 == 0) compare_u64(tag, A);
                ++count;
            }
        }
    }
    return count;
}

// ---------------------------------------------------------------------------
// Part 1c: a whole compute_cohomology() at ell = 3, route on vs off
// ---------------------------------------------------------------------------
template<class M>
bool same_fm_lists(const std::vector<std::vector<M>>& a, const std::vector<std::vector<M>>& b) {
    if (a.size() != b.size()) return false;
    for (size_t q = 0; q < a.size(); ++q) {
        if (a[q].size() != b[q].size()) return false;
        for (size_t t = 0; t < a[q].size(); ++t) {
            const M& x = a[q][t];
            const M& y = b[q][t];
            if (x.get_rows() != y.get_rows() || x.get_cols() != y.get_cols()) return false;
            for (uint32_t i = 0; i < x.get_rows(); ++i)
                for (uint32_t j = 0; j < x.get_cols(); ++j)
                    if (x.get_entry(i, j) != y.get_entry(i, j)) return false;
        }
    }
    return true;
}
void pipeline_part(uint32_t n_sym) {
    const SimplicialComplex K = CreateSL2ZFundamentalDomainTriangulation();
    const LocalSystemSimplicialComplex lssc(K, relations(K, n_sym), n_sym + 1, 3u);
    const std::string tag = "compute_cohomology ell=3 n_sym=" + str(n_sym);
    EquivariantCohomologyOnK off(&lssc), on(&lssc);
    std::string err_off, err_on;
    {
        elim::ScopedTernaryF3 g(false);
        try { off.compute_cohomology(); } catch (const std::exception& e) { err_off = e.what(); }
    }
    const elim::TernaryF3Totals t0 = elim::ternary_f3_totals();
    {
        elim::ScopedTernaryF3 g(true);
        try { on.compute_cohomology(); } catch (const std::exception& e) { err_on = e.what(); }
    }
    const elim::TernaryF3Totals t1 = elim::ternary_f3_totals();
    check(err_off == err_on, tag + ": same outcome (" + (err_off.empty() ? "ok" : err_off) + ")");
    if (!err_off.empty() || !err_on.empty()) return;
    check(off.get_H_eq_dimensions() == on.get_H_eq_dimensions(), tag + ": dim H^q");
    {
        elim::ScopedTernaryF3 g(false);
        check(same_fm_lists(off.get_Z_eq_bases_fm(), on.get_Z_eq_bases_fm()), tag + ": Z_eq bases");
        check(same_fm_lists(off.get_B_eq_bases_fm(), on.get_B_eq_bases_fm()), tag + ": B_eq bases");
        check(same_fm_lists(off.get_H_eq_bases_fm(), on.get_H_eq_bases_fm()), tag + ": H_eq bases");
    }
    const uint64_t routed = (t1.rrefs - t0.rrefs) + (t1.rank_profiles - t0.rank_profiles);
    std::cout << "  " << tag << ": dims";
    for (uint32_t d : on.get_H_eq_dimensions()) std::cout << " " << d;
    std::cout << ", routed eliminations " << routed << std::endl;
    check(routed > 0, tag + ": the route ran inside the pipeline");
}

// ---------------------------------------------------------------------------
// Part 2: the reader's F_3 route against its Ops3<1> route
// ---------------------------------------------------------------------------
std::string cell_str(const l23::ModCell& c) {
    std::string s = "h0=" + str(c.h0) + " h1=" + str(c.length) + " img=";
    for (uint32_t x : c.image_length) s += str(x) + ",";
    return s;
}
bool same_cell(const l23::ModCell& a, const l23::ModCell& b) {
    return a.ell == b.ell && a.n == b.n && a.k == b.k && a.h0 == b.h0 && a.length == b.length && a.image_length == b.image_length;
}
uint32_t reader_part() {
    const std::vector<std::pair<uint32_t, int64_t>> ops = {{3, 0}, {3, 1}, {2, 3}, {5, 6}};
    // (TER-1 audit F1) even n only: at odd n every recorded value is 0 whatever rho(T) computes.  Around 3^k +- 1 the
    // top level of the characteristic-3 shift is barely entered or exactly full (d = 3^k, 3^k +- 1, 3^k + 2).
    std::vector<uint32_t> ns;
    for (uint32_t n = 0; n <= 40; n += 2) ns.push_back(n);
    for (uint32_t n : {52u, 64u, 78u, 80u, 82u, 100u, 128u, 160u, 240u, 242u, 244u}) ns.push_back(n);
    uint32_t count = 0;
    for (uint32_t n : ns) {
        for (uint32_t mutate : {0u, 1u}) {
            if (mutate && n % 4 != 2) continue;
            l23::ModCell a, b;
            std::string ea, eb;
            { l23::ScopedTernaryF3Reader off(false); try { a = l23::h1_mod(3, n, 1, ops, mutate); } catch (const std::exception& e) { ea = e.what(); } }
            { l23::ScopedTernaryF3Reader on(true);   try { b = l23::h1_mod(3, n, 1, ops, mutate); } catch (const std::exception& e) { eb = e.what(); } }
            const std::string tag = "h1_mod(3, " + str(n) + ", 1) mutate=" + str(mutate);
            check(ea == eb, tag + ": same outcome (" + (ea.empty() ? "ok" : ea) + " / " + (eb.empty() ? "ok" : eb) + ")");
            if (ea.empty() && eb.empty())
                check(same_cell(a, b), tag + ": F_3 route == Ops3<1> route (" + cell_str(b) + " vs " + cell_str(a) + ")");
            ++count;
        }
    }
    {   // (TER-1 audit F1) the shift itself: Rho<Ops3F>::T / Ti vs the d^2 / 2 row-operation shift, entrywise, and T^-1 T = id
        std::vector<uint32_t> ds;
        for (uint32_t d = 1; d <= 300; ++d) ds.push_back(d);
        for (uint32_t p3 = 3; p3 <= 729; p3 *= 3)
            for (int e = -2; e <= 2; ++e) if (p3 + e > 300) ds.push_back(static_cast<uint32_t>(static_cast<int>(p3) + e));
        uint64_t bad = 0;
        for (uint32_t d : ds) bad += l23::taylor3_selfcheck(d - 1, d <= 64 ? 37u : 9u, 0x7A4ULL + d);
        check(bad == 0, "taylor3 == the d^2/2 shift (T and Ti, entrywise) and Ti T = id on " + str(ds.size()) + " sizes d in 1..731 (" + str(bad) + " differing)");
        std::cout << "  TAYLOR3_SELFCHECK sizes=" << ds.size() << " d_max=" << ds.back() << " differing=" << bad << std::endl;
    }
    {   // the default follows L23_TERNARY_F3 (on unless "0"); k >= 2 never takes the route
        l23::ScopedTernaryF3Reader on(true);
        const l23::ModCell c2 = l23::h1_mod(3, 12, 2, {{3, 0}});
        l23::ScopedTernaryF3Reader off(false);
        const l23::ModCell d2 = l23::h1_mod(3, 12, 2, {{3, 0}});
        check(same_cell(c2, d2), "h1_mod(3, 12, 2): k = 2 unaffected by the F_3 switch");
    }
    return count;
}

// ---------------------------------------------------------------------------
// Part 3: timings (reported, never gated)
// ---------------------------------------------------------------------------
// TritMatrix::rref (best of 3) and the routed elim::rref against the engine at p = 3 on uint32_t (both sizes) and
// uint64_t (n <= 1024 only: the engine's uint64_t field route reduces through a 128-bit division and takes about a
// minute at 4096^2 -- the reader's Ops3<1> route calls exactly that instantiation).
void timing(uint32_t n) {
    uint64_t s = 0x7173ULL + n;
    const DenseBuf<uint32_t> A = random_dense(n, n, 170, s);
    const bin::TritMatrix T0 = bin::TritMatrix::from_dense<uint32_t>(cview(A));
    double best = 1e30;
    bin::TritRref tr;
    bin::TritMatrix R;
    for (int rep = 0; rep < 3; ++rep) {
        R = T0;
        const double t = now_s();
        tr = bin::rref(R);
        best = std::min(best, now_s() - t);
    }
    double t_route, t_eng;
    DenseBuf<uint32_t> Rd = A, Re = A;
    elim::RrefResult rd, re;
    { elim::ScopedTernaryF3 on(true);   const double t = now_s(); rd = elim::rref<uint32_t>(ctx3(), view(Rd)); t_route = now_s() - t; }
    { elim::ScopedTernaryF3 off(false); const double t = now_s(); re = elim::rref<uint32_t>(ctx3(), view(Re)); t_eng = now_s() - t; }
    check(tr.pivots == re.pivots && rd.pivots == re.pivots && Rd.same_entries(Re) && unpack(R).same_entries(Re),
          "timing n=" + str(n) + ": trit, routed and engine RREF agree");
    double t_eng64 = -1, t_route64 = -1;
    if (n <= 1024) {
        const engine::ModCtx<uint64_t> c64(3u);
        DenseBuf<uint64_t> A64(n, n);
        for (uint32_t i = 0; i < n; ++i)
            for (uint32_t j = 0; j < n; ++j) A64.at(i, j) = A.at(i, j);
        DenseBuf<uint64_t> Rd64 = A64, Re64 = A64;
        elim::RrefResult rd64, re64;
        { elim::ScopedTernaryF3 on(true);   const double t = now_s(); rd64 = elim::rref<uint64_t>(c64, view(Rd64)); t_route64 = now_s() - t; }
        { elim::ScopedTernaryF3 off(false); const double t = now_s(); re64 = elim::rref<uint64_t>(c64, view(Re64)); t_eng64 = now_s() - t; }
        check(rd64.pivots == re.pivots && re64.pivots == re.pivots && Rd64.same_entries(Re64), "timing n=" + str(n) + ": uint64_t routed == engine");
    }
    char buf[640];
    std::snprintf(buf, sizeof(buf),
                  "TERNARY_F3_TIMING n=%u body=%s rank=%u trit_rref_s=%.4f routed_u32_s=%.4f engine_u32_s=%.4f speedup_trit_u32=%.1fx "
                  "speedup_routed_u32=%.1fx",
                  n, bin::ternary_row_body_name(), tr.rank, best, t_route, t_eng, t_eng / best, t_eng / t_route);
    std::string line = buf;
    if (t_eng64 >= 0) {
        std::snprintf(buf, sizeof(buf), " routed_u64_s=%.4f engine_u64_s=%.4f speedup_routed_u64=%.1fx", t_route64, t_eng64, t_eng64 / t_route64);
        line += buf;
    }
    std::cout << line << std::endl;
}
void reader_timing(uint32_t n) {
    const std::vector<std::pair<uint32_t, int64_t>> ops = {{3, 0}, {3, 1}};
    l23::ModCell a, b;
    double ta, tb;
    { l23::ScopedTernaryF3Reader off(false); const double t = now_s(); a = l23::h1_mod(3, n, 1, ops); ta = now_s() - t; }
    { l23::ScopedTernaryF3Reader on(true);   const double t = now_s(); b = l23::h1_mod(3, n, 1, ops); tb = now_s() - t; }
    check(same_cell(a, b), "reader timing n=" + str(n) + ": F_3 route == Ops3<1> route");
    char buf[512];
    std::snprintf(buf, sizeof(buf), "TERNARY_F3_READER ell=3 n=%u k=1 %s ops3_s=%.3f f3_s=%.3f speedup=%.1fx", n, cell_str(b).c_str(), ta, tb, ta / tb);
    std::cout << buf << std::endl;
}
} // namespace

uint32_t TernaryF3Validation() {
    const double t0 = now_s();
    g_fail = 0; g_checks = 0;
    std::cout << "=== TernaryF3Validation (engine::binary TritMatrix, the ENGINE_TERNARY_F3 route, the reader's F_3 route) ==="
              << std::endl;
    std::string paths;
    for (bin::RowPath p : available_paths()) paths += std::string(paths.empty() ? "" : ",") + bin::row_path_name(p);
    std::cout << "  row paths available: " << paths << " (auto = " << bin::row_path_name(bin::active_row_path()) << ", body "
              << bin::ternary_row_body_name() << "); ENGINE_TERNARY_F3 default route " << (elim::ternary_f3_enabled() ? "ON (env)" : "off")
              << "; reader F_3 route " << (l23::ternary_f3_reader_enabled() ? "on (default)" : "OFF (L23_TERNARY_F3=0)") << std::endl;
    uint32_t n_random = 0, n_amalgam = 0, n_reader = 0;
    try {
        {   // negative controls: p != 3 never takes the route; a non-F_3 entry is refused
            elim::ScopedTernaryF3 on(true);
            const engine::ModCtx<uint32_t> c5(5u);
            DenseBuf<uint32_t> A(4, 4);
            for (uint32_t i = 0; i < 4; ++i) A.at(i, (i + 1) % 4) = 4;
            const elim::TernaryF3Totals a = elim::ternary_f3_totals();
            (void)elim::rref<uint32_t>(c5, view(A));
            const elim::TernaryF3Totals b = elim::ternary_f3_totals();
            check(a.rrefs == b.rrefs, "negative control: p = 5 does not take the F_3 route");
            bool threw = false;
            DenseBuf<uint32_t> bad(2, 2);
            bad.at(0, 0) = 3;
            try { (void)bin::TritMatrix::from_dense<uint32_t>(cview(bad)); } catch (const std::logic_error&) { threw = true; }
            check(threw, "negative control: from_dense refuses a non-F_3 entry");
        }
        arithmetic_part();
        n_random = random_part();
        std::cout << "  random matrices: " << n_random << " (" << g_checks << " checks so far, " << g_fail << " failing)" << std::endl;
        n_amalgam = amalgam_part();
        std::cout << "  amalgam (A_Cq, 7-vertex SL_2(Z) model mod 3) matrices: " << n_amalgam << " (" << g_checks << " checks so far, "
                  << g_fail << " failing)" << std::endl;
        check(n_amalgam >= 100, "at least 100 amalgam-shaped matrices");
        pipeline_part(16);
        pipeline_part(18);
        n_reader = reader_part();
        std::cout << "  reader cells (h1_mod ell = 3, k = 1, F_3 route vs Ops3<1>): " << n_reader << " (" << g_checks
                  << " checks so far, " << g_fail << " failing)" << std::endl;
        timing(1024);
        timing(4096);
        reader_timing(256);
    } catch (const std::exception& e) {
        check(false, std::string("exception: ") + e.what());
    }
    bin::set_row_path(bin::RowPath::Auto);
    const double secs = now_s() - t0;
    std::cout << "TERNARY_F3 random=" << n_random << " amalgam=" << n_amalgam << " reader_cells=" << n_reader
              << " paths=" << paths << " checks=" << g_checks << " failing=" << g_fail << " verdict=" << (g_fail == 0 ? "OK" : "FAIL")
              << std::endl;
    std::cout << "  " << secs << " s" << std::endl;
    std::cout << (g_fail == 0 ? "TernaryF3Validation: PASS" : "TernaryF3Validation: FAIL") << std::endl;
    return g_fail;
}
