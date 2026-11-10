// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

#include "BinaryF2Validation.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "BitMatrix.h"
#include "../ModArithmetic/DenseBuf.h"
#include "../ModArithmetic/DenseElim.h"
#include "../ModArithmetic/DenseView.h"
#include "../ModArithmetic/FlatMatrix.h"
#include "../ModArithmetic/Kernels.h"
#include "../ModArithmetic/ModCtx.h"
#include "../ModArithmetic/RefLinAlg.h"
#include "../../hacs/EquivariantCohomologyOnK.h"
#include "../../hacs/LocalSystemSL2Z.h"
#include "../../hacs/LocalSystemSimplicialComplex.h"
#include "../../hacs/SimplicialCohomology.h"

// See BinaryF2Validation.h.

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

const engine::ModCtx<uint32_t>& ctx2() { static const engine::ModCtx<uint32_t> c(2u); return c; }

// density in 1/256 units: an entry is 1 with probability dens / 256.
DenseBuf<uint32_t> random_dense(uint32_t m, uint32_t n, uint32_t dens, uint64_t& s) {
    DenseBuf<uint32_t> A(m, n);
    for (uint32_t i = 0; i < m; ++i)
        for (uint32_t j = 0; j < n; ++j) A.at(i, j) = (splitmix(s) & 255u) < dens ? 1u : 0u;
    return A;
}
// L (m x r) * R (r x n) by the engine's gemm at p = 2: rank <= r.
DenseBuf<uint32_t> low_rank(uint32_t m, uint32_t n, uint32_t r, uint64_t& s) {
    const DenseBuf<uint32_t> L = random_dense(m, r, 128, s), R = random_dense(r, n, 128, s);
    DenseBuf<uint32_t> C(m, n);
    engine::kernels::gemm<uint32_t>(ctx2(), cview(L), cview(R), view(C), engine::kernels::GemmOp::Set);
    return C;
}

DenseBuf<uint32_t> unpack(const bin::BitMatrix& B) {
    DenseBuf<uint32_t> D(B.rows(), B.cols());
    B.to_dense<uint32_t>(view(D));
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

// The full comparison of one matrix (ENGINE_API.md Sec. 8.1's contract).
void compare_one(const std::string& tag, const DenseBuf<uint32_t>& A, bool with_oracle, bool with_product,
                 uint64_t& s) {
    const auto& ctx = ctx2();
    // (1) the engine at p = 2, route forced off
    DenseBuf<uint32_t> Re = A;
    elim::RrefResult re;
    DenseBuf<uint32_t> Ke;
    elim::RrefResult crp_e;
    {
        elim::ScopedBinaryF2 off(false);
        re = elim::rref<uint32_t>(ctx, view(Re));
        Ke = elim::kernel_basis<uint32_t>(ctx, cview(Re), re.pivots);
        DenseBuf<uint32_t> tmp = A;
        crp_e = elim::column_rank_profile<uint32_t>(ctx, view(tmp));
    }
    check(crp_e.pivots == re.pivots, tag + ": engine column_rank_profile == rref pivots");

    // (2) packing round trip
    const bin::BitMatrix B = bin::BitMatrix::from_dense<uint32_t>(cview(A));
    check(unpack(B).same_entries(A), tag + ": pack / unpack round trip");

    // (3) every row path: rref, echelon, rank, kernel
    uint64_t fnv_auto = 0;
    {
        PathGuard g;
        bool first = true;
        for (bin::RowPath p : available_paths()) {
            bin::set_row_path(p);
            const std::string pt = tag + " [" + bin::row_path_name(p) + "]";
            // (BIN-1 audit R1 / F1) the pack on THIS path: from_dense<uint32_t> picks its body by
            // the active row path, so every path's pack is exercised on any machine.
            check(bin::BitMatrix::from_dense<uint32_t>(cview(A)) == B, pt + ": from_dense on this path == the Auto pack");
            bin::BitMatrix R = B;
            const bin::BitRref br = bin::rref(R);
            check(br.rank == re.rank && br.pivots == re.pivots, pt + ": rank / pivots == engine");
            if (first) {
                check(unpack(R).same_entries(Re), pt + ": RREF entrywise == engine");
                fnv_auto = R.fnv();
                const bin::BitMatrix K = bin::kernel_basis(R, br.pivots);
                check(K.rows() == Ke.rows && unpack(K).same_entries(Ke), pt + ": kernel basis entrywise == engine");
                check(bin::multiply_naive(B, [&] {   // A * K^T == 0
                          bin::BitMatrix Kt(K.cols(), K.rows());
                          for (uint32_t i = 0; i < K.rows(); ++i)
                              for (uint32_t j = 0; j < K.cols(); ++j) if (K.get(i, j)) Kt.set(j, i, true);
                          return Kt;
                      }()).popcount() == 0, pt + ": A * kernel^T == 0");
                first = false;
            } else {
                check(R.fnv() == fnv_auto, pt + ": RREF bits == the first path's");
            }
            bin::BitMatrix E = B;
            const bin::BitRref be = bin::echelon(E);
            check(be.rank == re.rank && be.pivots == re.pivots, pt + ": echelon pivots == engine");
            check(bin::rank(B) == re.rank, pt + ": rank() == engine");
        }
    }
    // (3b) (BIN-1 audit F2) the general-stride pack / unpack, through a transposed view (column stride != 1)
    if (with_product) {
        const bin::BitMatrix Bt = bin::BitMatrix::from_dense<uint32_t>(cview(A).transposed_view());
        bool ok = Bt.rows() == A.cols && Bt.cols() == A.rows;
        for (uint32_t i = 0; ok && i < A.rows; ++i)
            for (uint32_t j = 0; ok && j < A.cols; ++j) ok = Bt.get(j, i) == (A.at(i, j) != 0);
        DenseBuf<uint32_t> back(A.rows, A.cols);
        Bt.to_dense<uint32_t>(view(back).transposed_view());
        check(ok && back.same_entries(A), tag + ": strided (transposed-view) pack / unpack round trip");
    }
    // (4) the naive reference elimination
    {
        bin::BitMatrix R = B;
        const bin::BitRref bn = bin::rref_naive(R);
        check(bn.pivots == re.pivots && R.fnv() == fnv_auto, tag + ": rref_naive == M4RI");
    }
    // (5) the dispatched route of engine::elim
    {
        elim::ScopedBinaryF2 on(true);
        const elim::BinaryF2Totals t0 = elim::binary_f2_totals();
        DenseBuf<uint32_t> Rd = A;
        const elim::RrefResult rd = elim::rref<uint32_t>(ctx, view(Rd));
        DenseBuf<uint32_t> tmp = A;
        const elim::RrefResult cd = elim::column_rank_profile<uint32_t>(ctx, view(tmp));
        const elim::BinaryF2Totals t1 = elim::binary_f2_totals();
        const bool nonempty = A.rows > 0 && A.cols > 0;
        check(rd.rank == re.rank && rd.pivots == re.pivots && Rd.same_entries(Re), tag + ": routed elim::rref == engine");
        check(cd.pivots == re.pivots && cd.rank == re.rank, tag + ": routed column_rank_profile == engine");
        check(!nonempty || (t1.rrefs == t0.rrefs + 1 && t1.rank_profiles == t0.rank_profiles + 1),
              tag + ": route counters advanced (the route really ran)");
    }
    // (6) the oracle
    if (with_oracle) {
        DenseBuf<uint32_t> Rr = A;
        const std::vector<uint32_t> pv = engine::ref::rref<uint32_t>(2u, view(Rr));
        check(pv == re.pivots && Rr.same_entries(Re), tag + ": RREF / pivots == RefLinAlg");
    }
    // (7) the product
    if (with_product) {
        const uint32_t k = 1 + static_cast<uint32_t>(splitmix(s) % 200);
        const DenseBuf<uint32_t> Y = random_dense(A.cols, k, 128, s);
        DenseBuf<uint32_t> Ce(A.rows, k);
        engine::kernels::gemm<uint32_t>(ctx, cview(A), cview(Y), view(Ce), engine::kernels::GemmOp::Set);
        const bin::BitMatrix Yb = bin::BitMatrix::from_dense<uint32_t>(cview(Y));
        PathGuard g;
        for (bin::RowPath p : available_paths()) {
            bin::set_row_path(p);
            check(bin::BitMatrix::from_dense<uint32_t>(cview(Y)) == Yb,
                  tag + " [" + bin::row_path_name(p) + "]: from_dense of the product operand on this path");   // R1
            const bin::BitMatrix C = bin::multiply_m4rm(B, Yb);
            check(unpack(C).same_entries(Ce), tag + " [" + bin::row_path_name(p) + "]: M4RM product == engine gemm");
        }
        check(unpack(bin::multiply_naive(B, Yb)).same_entries(Ce), tag + ": naive product == engine gemm");
    }
}

// The uint64_t instantiation of the route (rref only).
void compare_u64(const std::string& tag, const DenseBuf<uint32_t>& A32) {
    const engine::ModCtx<uint64_t> ctx(2u);
    DenseBuf<uint64_t> A(A32.rows, A32.cols);
    for (uint32_t i = 0; i < A.rows; ++i)
        for (uint32_t j = 0; j < A.cols; ++j) A.at(i, j) = A32.at(i, j);
    DenseBuf<uint64_t> Re = A, Rd = A;
    elim::RrefResult re, rd;
    { elim::ScopedBinaryF2 off(false); re = elim::rref<uint64_t>(ctx, view(Re)); }
    { elim::ScopedBinaryF2 on(true);   rd = elim::rref<uint64_t>(ctx, view(Rd)); }
    check(re.pivots == rd.pivots && re.rank == rd.rank && Re.same_entries(Rd), tag + ": uint64_t routed rref == engine");
}

// ---------------------------------------------------------------------------
// Part 1: 500 random matrices
// ---------------------------------------------------------------------------
uint32_t random_part() {
    uint64_t s = 0xB1A2F2ULL;
    const uint32_t dens_list[] = {128, 128, 26, 5, 230, 64};
    const uint32_t edge_n[] = {1, 63, 64, 65, 127, 128, 129, 511, 512, 513};
    uint32_t count = 0;
    for (uint32_t t = 0; t < 500; ++t) {
        uint32_t m, n;
        if (t < 20) {                                   // word / lane boundaries, both orientations
            m = (t & 1) ? 1 + static_cast<uint32_t>(splitmix(s) % 80) : edge_n[(t / 2) % 10];
            n = (t & 1) ? edge_n[(t / 2) % 10] : 1 + static_cast<uint32_t>(splitmix(s) % 80);
        } else if (t % 50 == 7) {                       // larger
            m = 300 + static_cast<uint32_t>(splitmix(s) % 800);
            n = 300 + static_cast<uint32_t>(splitmix(s) % 800);
        } else {
            m = 1 + static_cast<uint32_t>(splitmix(s) % 200);
            n = 1 + static_cast<uint32_t>(splitmix(s) % 200);
        }
        DenseBuf<uint32_t> A;
        std::string kind;
        if (t == 20) { A = DenseBuf<uint32_t>(m, n); kind = "zero"; }
        else if (t == 21) { A = DenseBuf<uint32_t>(n, n); for (uint32_t i = 0; i < n; ++i) A.at(i, i) = 1; kind = "identity"; }
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
        compare_one(tag, A, (t % 5 == 0) && static_cast<uint64_t>(m) * n <= 40000, t % 10 == 0, s);
        if (t % 25 == 0) compare_u64(tag, A);
        ++count;
    }
    return count;
}

// ---------------------------------------------------------------------------
// Part 2: the real A_Cq matrices of the 7-vertex SL_2(Z) model mod 2
// (assembled exactly as DenseElimPipelineValidation / compute_C_eq_bases do).
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
    for (const auto& datum : GenerateSL2ZRelationsWithGroupData(K, n_sym, 2u)) rels.push_back(datum.rel);
    return rels;
}

uint32_t amalgam_part() {
    uint64_t s = 0xA3A16AULL;
    const uint32_t n_list[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 16, 17, 24, 31, 32, 63, 64, 100, 128};
    const SimplicialComplex K = CreateSL2ZFundamentalDomainTriangulation();
    uint32_t count = 0;
    for (int sc = 1; sc >= 0; --sc) {
        EquivariantCohomologyOptions::ScopedVacuousShortcircuit guard(sc != 0);
        for (uint32_t n_sym : n_list) {
            const LocalSystemSimplicialComplex lssc(K, relations(K, n_sym), n_sym + 1, 2u);
            for (uint32_t q = 0; q <= 2; ++q) {
                const DenseBuf<uint32_t> A = build_ACq(lssc, q);
                if (A.rows == 0 || A.cols == 0) continue;
                const std::string tag = "A_C" + str(q) + " n_sym=" + str(n_sym) + (sc ? " sc=on " : " sc=off ") +
                                        str(A.rows) + "x" + str(A.cols);
                compare_one(tag, A, static_cast<uint64_t>(A.rows) * A.cols <= 40000, count % 7 == 0, s);
                if (count % 10 == 0) compare_u64(tag, A);
                ++count;
            }
        }
    }
    return count;
}

// ---------------------------------------------------------------------------
// Part 3: a whole compute_cohomology() at ell = 2, route on vs off
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
    const LocalSystemSimplicialComplex lssc(K, relations(K, n_sym), n_sym + 1, 2u);
    const std::string tag = "compute_cohomology ell=2 n_sym=" + str(n_sym);
    EquivariantCohomologyOnK off(&lssc), on(&lssc);
    std::string err_off, err_on;
    {
        elim::ScopedBinaryF2 g(false);
        try { off.compute_cohomology(); } catch (const std::exception& e) { err_off = e.what(); }
    }
    const elim::BinaryF2Totals t0 = elim::binary_f2_totals();
    {
        elim::ScopedBinaryF2 g(true);
        try { on.compute_cohomology(); } catch (const std::exception& e) { err_on = e.what(); }
    }
    const elim::BinaryF2Totals t1 = elim::binary_f2_totals();
    check(err_off == err_on, tag + ": same outcome (" + (err_off.empty() ? "ok" : err_off) + ")");
    if (!err_off.empty() || !err_on.empty()) return;
    check(off.get_H_eq_dimensions() == on.get_H_eq_dimensions(), tag + ": dim H^q");
    {
        elim::ScopedBinaryF2 g(false);   // the lazy accessors may eliminate: keep both off when materialising
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
// Part 4: timing (reported, never gated)
// ---------------------------------------------------------------------------
void timing(uint32_t n) {
    uint64_t s = 0x7173ULL + n;
    const DenseBuf<uint32_t> A = random_dense(n, n, 128, s);
    const bin::BitMatrix B0 = bin::BitMatrix::from_dense<uint32_t>(cview(A));
    double best = 1e30;
    bin::BitRref br;
    bin::BitMatrix R;
    for (int rep = 0; rep < 3; ++rep) {
        R = B0;
        const double t = now_s();
        br = bin::rref(R);
        best = std::min(best, now_s() - t);
    }
    double t_route;
    DenseBuf<uint32_t> Rd = A;
    elim::RrefResult rd;
    {
        elim::ScopedBinaryF2 on(true);
        const double t = now_s();
        rd = elim::rref<uint32_t>(ctx2(), view(Rd));
        t_route = now_s() - t;
    }
    DenseBuf<uint32_t> Re = A;
    elim::RrefResult re;
    double t_eng;
    {
        elim::ScopedBinaryF2 off(false);
        const double t = now_s();
        re = elim::rref<uint32_t>(ctx2(), view(Re));
        t_eng = now_s() - t;
    }
    check(br.pivots == re.pivots && rd.pivots == re.pivots && Rd.same_entries(Re) && unpack(R).same_entries(Re),
          "timing n=" + str(n) + ": binary, routed and engine RREF agree");
    double t_mm;
    {
        uint64_t s2 = 0x55ULL + n;
        const bin::BitMatrix Y = bin::BitMatrix::from_dense<uint32_t>(cview(random_dense(n, n, 128, s2)));
        const double t = now_s();
        const bin::BitMatrix C = bin::multiply_m4rm(B0, Y);
        t_mm = now_s() - t;
        check(C.rows() == n, "timing n=" + str(n) + ": product shape");
    }
    char buf[512];
    std::snprintf(buf, sizeof(buf),
                  "BINARY_F2_TIMING n=%u path=%s rank=%u bit_rref_s=%.4f routed_rref_s=%.4f engine_rref_s=%.4f "
                  "speedup_bit=%.1fx speedup_routed=%.1fx m4rm_s=%.4f%s",
                  n, bin::row_path_name(bin::active_row_path()), br.rank, best, t_route, t_eng, t_eng / best,
                  t_eng / t_route, t_mm, n == 4096 ? (best <= 0.06 ? " target_0.06s=MET" : " target_0.06s=MISSED") : "");
    std::cout << buf << std::endl;
}

} // namespace

uint32_t BinaryF2Validation() {
    const double t0 = now_s();
    g_fail = 0; g_checks = 0;
    std::cout << "=== BinaryF2Validation (engine::binary M4RI / M4RM and the ENGINE_BINARY_F2 route vs the engine at p = 2) ==="
              << std::endl;
    std::string paths;
    for (bin::RowPath p : available_paths()) paths += std::string(paths.empty() ? "" : ",") + bin::row_path_name(p);
    std::cout << "  row paths available: " << paths << " (auto = " << bin::row_path_name(bin::active_row_path())
              << "); ENGINE_BINARY_F2 default route " << (elim::binary_f2_enabled() ? "ON (env)" : "off") << std::endl;
    uint32_t n_random = 0, n_amalgam = 0;
    try {
        // Negative control: a context with p != 2 never takes the route.
        {
            elim::ScopedBinaryF2 on(true);
            const engine::ModCtx<uint32_t> c3(3u);
            DenseBuf<uint32_t> A(4, 4);
            for (uint32_t i = 0; i < 4; ++i) A.at(i, (i + 1) % 4) = 2;
            const elim::BinaryF2Totals a = elim::binary_f2_totals();
            (void)elim::rref<uint32_t>(c3, view(A));
            const elim::BinaryF2Totals b = elim::binary_f2_totals();
            check(a.rrefs == b.rrefs, "negative control: p = 3 does not take the F_2 route");
            bool threw = false;
            DenseBuf<uint32_t> bad(2, 2);
            bad.at(0, 0) = 3;
            try { (void)bin::BitMatrix::from_dense<uint32_t>(cview(bad)); } catch (const std::logic_error&) { threw = true; }
            check(threw, "negative control: from_dense refuses a non-F_2 entry");
        }
        n_random = random_part();
        std::cout << "  random matrices: " << n_random << " (" << g_checks << " checks so far, " << g_fail << " failing)" << std::endl;
        n_amalgam = amalgam_part();
        std::cout << "  amalgam (A_Cq, 7-vertex SL_2(Z) model mod 2) matrices: " << n_amalgam << " (" << g_checks
                  << " checks so far, " << g_fail << " failing)" << std::endl;
        check(n_amalgam >= 100, "at least 100 amalgam-shaped matrices");
        pipeline_part(16);
        pipeline_part(17);
        timing(1024);
        timing(4096);
    } catch (const std::exception& e) {
        check(false, std::string("exception: ") + e.what());
    }
    bin::set_row_path(bin::RowPath::Auto);
    const double secs = now_s() - t0;
    std::cout << "BINARY_F2 random=" << n_random << " amalgam=" << n_amalgam << " paths=" << paths << " checks=" << g_checks
              << " failing=" << g_fail << " verdict=" << (g_fail == 0 ? "OK" : "FAIL") << std::endl;
    std::cout << "  " << secs << " s" << std::endl;
    std::cout << (g_fail == 0 ? "BinaryF2Validation: PASS" : "BinaryF2Validation: FAIL") << std::endl;
    return g_fail;
}
