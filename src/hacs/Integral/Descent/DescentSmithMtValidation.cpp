// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// (slot-24 PAR-2K) DescentSmithMtValidation: the threaded ring2k / ring3k Smith against the serial code.  See the header.
// Read-only measurement hooks (they replace nothing in the arms; each prints an EFFECTIVE line):
//   HECKE_DESCENT_SMITH_MT_BENCH=<D>[,<D>...]       ring2k::smith<u32> (k = 30) and ring3k::smith (k = 19) on a random D x D
//                                                   matrix (a product of two random D x D/2+ factors plus a 2-power scaled
//                                                   copy: several valuation levels), timed at each thread count;
//   HECKE_DESCENT_SMITH_MT_CELLS=<N>:<n>[,...]      RunTreeLocalPieceN at ell = 2 (and 3 when the cell has nu_3 > 0... always run)
//                                                   on the level-N cell, timed at each thread count, values compared;
//   HECKE_DESCENT_SMITH_MT_THREADS=<t>[,<t>...]     the thread counts of the two hooks (default 1,2,4,8,16,32);
//   HECKE_DESCENT_SMITH_MT_REPEATS=<r>              repeats per thread count (default 1; the median is printed).

#include "DescentSmithMtValidation.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "DescentAssembly.h"
#include "DescentLevelN.h"
#include "DescentTree.h"
#include "../../../ModArithmetic/Binary/BitMatrix.h"
#include "../../../ModArithmetic/Binary/Ring2k.h"
#include "../../../ModArithmetic/Binary/Ring3k.h"
#include "../../../ModArithmetic/Binary/SmithPar.h"
#include "../../../ModArithmetic/Binary/TritMatrix.h"
#include "../../../ModArithmetic/Kernels.h"
#include "../../../ModArithmetic/DenseBuf.h"

namespace {

namespace r2 = engine::ring2k;
namespace r3 = engine::ring3k;
namespace sp = engine::smithpar;
using clk = std::chrono::steady_clock;

uint32_t g_checks = 0, g_fail = 0;
void check(bool ok, const std::string& what) {
    ++g_checks;
    if (!ok) {
        ++g_fail;
        std::cout << "  [FAIL] " << what << std::endl;
    }
}
double since(clk::time_point t) { return std::chrono::duration<double>(clk::now() - t).count(); }
std::string fx(double x, int p = 3) {
    std::ostringstream o;
    o << std::fixed << std::setprecision(p) << x;
    return o.str();
}

const std::vector<uint32_t> kThreads = {1, 2, 7, 16};

// ---- random ring2k inputs -------------------------------------------------------------------------------------------
template<class E>
E rnd_elt(std::mt19937_64& g) {
    using Tr = r2::Traits<E>;
    uint64_t w[r2::Traits<E>::kWords];
    for (uint32_t i = 0; i < Tr::kWords; ++i) w[i] = g();
    return Tr::from_words(w, Tr::kWords);
}

// kind 0: uniform; 1: a product B C of inner dimension r (rank-deficient over Z, several levels); 2: rows scaled by 2^(i mod 5).
template<class E>
r2::Mat<E> rnd_mat(std::mt19937_64& g, uint32_t m, uint32_t n, int kind) {
    using Tr = r2::Traits<E>;
    r2::Mat<E> A(m, n);
    if (kind == 1) {
        const uint32_t r = std::max<uint32_t>(1, std::min(m, n) * 2 / 3);
        r2::Mat<E> B(m, r), C(r, n);
        for (auto& x : B.a) x = Tr::shl(rnd_elt<E>(g), static_cast<uint32_t>(g() % 3));
        for (auto& x : C.a) x = Tr::shl(rnd_elt<E>(g), static_cast<uint32_t>(g() % 2));
        for (uint32_t i = 0; i < m; ++i)
            for (uint32_t t = 0; t < r; ++t) {
                const E f = B.at(i, t);
                for (uint32_t j = 0; j < n; ++j) A.at(i, j) = Tr::add(A.at(i, j), Tr::mul(f, C.at(t, j)));
            }
    } else {
        for (uint32_t i = 0; i < m; ++i)
            for (uint32_t j = 0; j < n; ++j) A.at(i, j) = kind == 2 ? Tr::shl(rnd_elt<E>(g), i % 5) : rnd_elt<E>(g);
    }
    return A;
}

template<class E>
bool same_mat(const r2::Mat<E>& a, const r2::Mat<E>& b) {
    if (a.rows != b.rows || a.cols != b.cols) return false;
    for (size_t i = 0; i < a.a.size(); ++i)
        if (!r2::Traits<E>::eq(a.a[i], b.a[i])) return false;
    return true;
}

bool same_info(const r2::EchelonInfo& a, const r2::EchelonInfo& b) {
    return a.rank == b.rank && a.pivots == b.pivots && a.pivot_v == b.pivot_v;
}

// One ring2k rung: every T in kThreads against T = 1 (the serial code), entrywise.
template<class E>
void arm_r2_rung(std::mt19937_64& g, uint32_t k, const std::vector<std::pair<uint32_t, uint32_t>>& shapes, const std::string& tag,
                 bool naive) {
    using Tr = r2::Traits<E>;
    for (const auto& [m, n] : shapes)
        for (int kind = 0; kind < 3; ++kind) {
            const r2::Mat<E> A0 = rnd_mat<E>(g, m, n, kind);
            const std::string t = tag + " " + std::to_string(m) + "x" + std::to_string(n) + " kind=" + std::to_string(kind);
            r2::Mat<E> Aser = A0, Aser2 = A0;
            r2::EchelonInfo eser, eser_odd;
            r2::SmithResult sser;
            r2::HowellForm<E> hser;
            {
                const sp::ScopedThreads st(1);
                eser = r2::echelon_blocked<Tr>(Aser, k, r2::PivotRule::MinValuation, r2::params().panel);
                sser = r2::smith<Tr>(A0, k);
                hser = r2::howell<Tr>(A0, k);
            }
            if (naive) check(r2::smith_naive<Tr>(A0, k).sigma == sser.sigma, t + ": smith == smith_naive");
            // the FirstOdd LU as smith() runs it: the F_2 rank profile first, col_limit = the F_2 rank
            uint32_t r0 = 0;
            {
                engine::binary::BitMatrix B(m, n);
                for (uint32_t i = 0; i < m; ++i)
                    for (uint32_t j = 0; j < n; ++j) if (Tr::odd(A0.at(i, j))) B.set(i, j, true);
                const engine::binary::BitRref br = engine::binary::echelon(B);
                r0 = br.rank;
                std::vector<uint32_t> order(br.pivots.begin(), br.pivots.end());
                for (uint32_t j = 0, q = 0; j < n; ++j) {
                    if (q < br.pivots.size() && br.pivots[q] == j) { ++q; continue; }
                    order.push_back(j);
                }
                for (uint32_t i = 0; i < m; ++i)
                    for (uint32_t j = 0; j < n; ++j) Aser2.at(i, j) = A0.at(i, order[j]);
            }
            r2::Mat<E> Aodd0 = Aser2;
            if (r0 > 0) {
                const sp::ScopedThreads st(1);
                eser_odd = r2::echelon_blocked<Tr>(Aser2, k, r2::PivotRule::FirstOdd, r2::params().panel, r0);
            }
            for (uint32_t T : kThreads) {
                if (T == 1) continue;
                const sp::ScopedThreads st(T);
                r2::Mat<E> A = A0;
                const r2::EchelonInfo e = r2::echelon_blocked<Tr>(A, k, r2::PivotRule::MinValuation, r2::params().panel);
                check(same_info(e, eser) && same_mat(A, Aser), t + " T=" + std::to_string(T) + ": MinValuation LU == serial, entrywise");
                if (r0 > 0) {
                    r2::Mat<E> Ao = Aodd0;
                    const r2::EchelonInfo eo = r2::echelon_blocked<Tr>(Ao, k, r2::PivotRule::FirstOdd, r2::params().panel, r0);
                    check(same_info(eo, eser_odd) && same_mat(Ao, Aser2), t + " T=" + std::to_string(T) + ": FirstOdd LU == serial, entrywise");
                }
                const r2::SmithResult s = r2::smith<Tr>(A0, k);
                check(s.sigma == sser.sigma && s.level_ranks == sser.level_ranks, t + " T=" + std::to_string(T) + ": smith == serial");
                const r2::HowellForm<E> h = r2::howell<Tr>(A0, k);
                check(h.pivots == hser.pivots && h.pivot_v == hser.pivot_v && same_mat(h.H, hser.H),
                      t + " T=" + std::to_string(T) + ": howell == serial, entrywise");
            }
        }
}

void arm_r2() {
    std::mt19937_64 g(0x2b7e151628aed2a6ull);
    const std::vector<std::pair<uint32_t, uint32_t>> shapes = {{37, 41}, {150, 130}, {261, 300}};
    const std::vector<std::pair<uint32_t, uint32_t>> small = {{37, 41}, {96, 80}};
    std::vector<r2::KernelPath> paths;
    for (r2::KernelPath p : {r2::KernelPath::Portable, r2::KernelPath::Avx2, r2::KernelPath::Avx512})
        if (r2::kernel_path_available(p)) paths.push_back(p);
    const r2::KernelPath saved = r2::active_kernel_path();
    std::string names;
    for (r2::KernelPath p : paths) {
        r2::set_kernel_path(p);
        const std::string tag = std::string("(R2) ") + r2::kernel_path_name(p);
        names += std::string(names.empty() ? "" : ",") + r2::kernel_path_name(p);
        arm_r2_rung<uint32_t>(g, 30, shapes, tag + " u32 k=30", p == paths.front());
        arm_r2_rung<uint64_t>(g, 61, shapes, tag + " u64 k=61", false);
        arm_r2_rung<uint16_t>(g, 13, small, tag + " u16 k=13", false);
        arm_r2_rung<uint8_t>(g, 8, small, tag + " u8 k=8", false);
        arm_r2_rung<r2::u128>(g, 100, small, tag + " u128 k=100", false);
        arm_r2_rung<r2::WordN<4>>(g, 200, small, tag + " WordN<4> k=200", false);
    }
    r2::set_kernel_path(r2::KernelPath::Auto);
    (void)saved;
    std::cout << "  (R2) ring2k: echelon_blocked (both rules) / smith / howell at T = 1, 2, 7, 16 == serial entrywise; paths " << names
              << "; rungs u8 u16 u32 u64 u128 WordN<4>" << std::endl;
}

// ---- ring3k ----------------------------------------------------------------------------------------------------------
engine::DenseBuf<uint32_t> rnd3(std::mt19937_64& g, uint32_t m, uint32_t n, uint32_t k, int kind) {
    uint32_t M = 1;
    for (uint32_t i = 0; i < k; ++i) M *= 3u;
    engine::DenseBuf<uint32_t> A(m, n);
    if (kind == 1) {
        const uint32_t r = std::max<uint32_t>(1, std::min(m, n) * 2 / 3);
        std::vector<uint64_t> B(static_cast<size_t>(m) * r), C(static_cast<size_t>(r) * n);
        for (auto& x : B) x = (g() % M) * ((g() % 3) == 0 ? 3u : 1u) % M;
        for (auto& x : C) x = (g() % M) * ((g() % 4) == 0 ? 9u : 1u) % M;
        for (uint32_t i = 0; i < m; ++i)
            for (uint32_t j = 0; j < n; ++j) {
                uint64_t s = 0;
                for (uint32_t t = 0; t < r; ++t) s = (s + B[static_cast<size_t>(i) * r + t] * C[static_cast<size_t>(t) * n + j]) % M;
                A.at(i, j) = static_cast<uint32_t>(s);
            }
    } else {
        uint32_t p = 1;
        for (uint32_t i = 0; i < m; ++i) {
            if (kind == 2) p = (i % 4 == 0) ? 1u : p * 3u % M;
            for (uint32_t j = 0; j < n; ++j) A.at(i, j) = static_cast<uint32_t>((g() % M) * (kind == 2 ? p : 1u) % M);
        }
    }
    return A;
}

bool same_buf(const engine::DenseBuf<uint32_t>& a, const engine::DenseBuf<uint32_t>& b) {
    if (a.rows != b.rows || a.cols != b.cols) return false;
    for (uint32_t i = 0; i < a.rows; ++i)
        for (uint32_t j = 0; j < a.cols; ++j)
            if (a.at(i, j) != b.at(i, j)) return false;
    return true;
}

void arm_r3() {
    std::mt19937_64 g(0x3243f6a8885a308dull);
    std::vector<engine::kernels::KernelPath> paths = {engine::kernels::KernelPath::Portable};
    if (engine::kernels::avx512_kernel_available()) paths.push_back(engine::kernels::KernelPath::Avx512);
    const engine::kernels::KernelPath saved = engine::kernels::kernel_path();
    std::string names;
    for (auto p : paths) {
        engine::kernels::set_kernel_path(p);
        const std::string pn = p == engine::kernels::KernelPath::Portable ? "portable" : "avx512";
        names += (names.empty() ? "" : ",") + pn;
        for (uint32_t k : {19u, 7u})
            for (const auto& [m, n] : std::vector<std::pair<uint32_t, uint32_t>>{{45, 39}, {170, 150}, {240, 270}})
                for (int kind = 0; kind < 3; ++kind) {
                    const engine::DenseBuf<uint32_t> A0 = rnd3(g, m, n, k, kind);
                    const std::string t = "(R3) " + pn + " k=" + std::to_string(k) + " " + std::to_string(m) + "x" + std::to_string(n) +
                                          " kind=" + std::to_string(kind);
                    r3::SmithResult sser;
                    r3::HowellForm hser;
                    {
                        const sp::ScopedThreads st(1);
                        sser = r3::smith_levels(A0, k);
                        hser = r3::howell_blocked(A0, k);
                    }
                    if (m <= 170 && p == paths.front()) check(r3::smith_naive(A0, k).sigma == sser.sigma, t + ": smith_levels == smith_naive");
                    for (uint32_t T : kThreads) {
                        if (T == 1) continue;
                        const sp::ScopedThreads st(T);
                        const r3::SmithResult s = r3::smith_levels(A0, k);
                        check(s.sigma == sser.sigma && s.level_ranks == sser.level_ranks, t + " T=" + std::to_string(T) + ": smith_levels == serial");
                        const r3::HowellForm h = r3::howell_blocked(A0, k);
                        check(h.pivots == hser.pivots && h.pivot_v == hser.pivot_v && same_buf(h.H, hser.H),
                              t + " T=" + std::to_string(T) + ": howell_blocked == serial, entrywise");
                    }
                }
    }
    engine::kernels::set_kernel_path(saved);
    std::cout << "  (R3) ring3k k = 19, 7: smith_levels / howell_blocked at T = 1, 2, 7, 16 == serial entrywise; engine GEMM paths "
              << names << std::endl;
}

// ---- real descent inputs ---------------------------------------------------------------------------------------------
bool same_piece(const integral_descent::LocalPiece& a, const integral_descent::LocalPiece& b) {
    if (a.d.size() != b.d.size() || a.tors.size() != b.tors.size() || a.all_certified() != b.all_certified()) return false;
    for (size_t q = 0; q < a.d.size(); ++q) {
        const auto &x = a.d[q], &y = b.d[q];
        if (x.rows != y.rows || x.cols != y.cols || x.r_K != y.r_K || x.rank_Q != y.rank_Q || x.W != y.W || x.nonunit != y.nonunit ||
            x.certified != y.certified)
            return false;
    }
    for (size_t q = 0; q < a.tors.size(); ++q)
        if (!(a.tors[q] == b.tors[q])) return false;
    return true;
}

std::vector<uint32_t> betti(uint32_t N, uint32_t n) {
    const auto c = integral_descent::LevelNClosedForm(N, n);
    std::vector<uint32_t> h(8, 0);
    h[0] = c.b0;
    h[1] = c.b1;
    return h;
}

void arm_d() {
    const sp::Stats s0 = sp::stats();
    uint32_t cells = 0;
    for (const auto& [N, n] : std::vector<std::pair<uint32_t, uint32_t>>{{11, 6}, {13, 10}, {25, 4}, {9, 8}, {7, 20}}) {
        const auto h = betti(N, n);
        for (uint32_t ell : {2u, 3u}) {
            const uint32_t q_top = ell == 2 ? 4u : 6u;
            integral_descent::LocalPiece ser;
            {
                const sp::ScopedThreads st(1);
                ser = integral_descent::RunTreeLocalPieceN(ell, N, n, q_top, h);
            }
            for (uint32_t T : kThreads) {
                if (T == 1) continue;
                const sp::ScopedThreads st(T);
                const auto par = integral_descent::RunTreeLocalPieceN(ell, N, n, q_top, h);
                check(same_piece(par, ser), "(D) tree ell=" + std::to_string(ell) + " (" + std::to_string(N) + "," + std::to_string(n) +
                                                ") T=" + std::to_string(T) + ": every differential and the torsion == serial");
            }
            ++cells;
        }
    }
    // the record route (AssembleDescentN) on the tree and on the dense rollback (DescentAssembly's Smith call site)
    for (bool tree : {true, false}) {
        const integral_descent::ScopedDescentTree sdt(tree);
        std::vector<integral_model::AbelianGroupType> Hser;
        {
            const sp::ScopedThreads st(1);
            const auto R = integral_descent::AssembleDescentN(11, 6);
            check(!R.refused, std::string("(D) AssembleDescentN(11, 6) ") + (tree ? "tree" : "dense") + " serial: not refused");
            Hser = R.H;
        }
        for (uint32_t T : {7u, 16u}) {
            const sp::ScopedThreads st(T);
            const auto R = integral_descent::AssembleDescentN(11, 6);
            bool same = !R.refused && R.H.size() == Hser.size();
            for (size_t q = 0; same && q < Hser.size(); ++q) same = R.H[q] == Hser[q];
            check(same, std::string("(D) AssembleDescentN(11, 6) ") + (tree ? "tree" : "dense") + " T=" + std::to_string(T) + ": H^0..H^5 == serial");
        }
    }
    const sp::Stats s1 = sp::stats();
    check(s1.regions_parallel > s0.regions_parallel, "(D) the threaded arms ran parallel regions (non-vacuous)");
    std::cout << "  (D) real inputs: RunTreeLocalPieceN on " << cells << " (cell, ell) pairs and AssembleDescentN(11, 6) tree + dense at T = 2, 7, 16"
              << " == serial; parallel regions " << (s1.regions_parallel - s0.regions_parallel) << std::endl;
}

// ---- (E) PAR-UNIFY (audit PAR2K L2): edge shapes, k << W, 64 threads, the panel 64 vs 128, the shared pool -------------
// kind 0 uniform; 1 all zero; 2 high valuation (multiples of 2^(k-1)) with ties (two repeated values); 3 (k < W only) entries
// = 0 mod 2^k with garbage above bit k, plus a sparse pattern of units (exercises the zero_mod skip); 4 duplicate rows.
template<class E>
r2::Mat<E> edge_mat(std::mt19937_64& g, uint32_t m, uint32_t n, uint32_t k, int kind) {
    using Tr = r2::Traits<E>;
    r2::Mat<E> A(m, n);
    const E t0 = Tr::shl(rnd_elt<E>(g), k - 1), t1 = Tr::shl(rnd_elt<E>(g), k - 1);
    for (uint32_t i = 0; i < m; ++i)
        for (uint32_t j = 0; j < n; ++j) {
            E x = Tr::zero();
            switch (kind) {
                case 0: x = rnd_elt<E>(g); break;
                case 1: break;
                case 2: x = (g() % 3 == 0) ? Tr::shl(rnd_elt<E>(g), k - 1) : ((g() & 1) ? t0 : t1); break;
                case 3: x = (g() % 23 == 0) ? rnd_elt<E>(g) : Tr::shl(rnd_elt<E>(g), k); break;
                default: x = (i > 0 && i % 3 == 0) ? A.at(i - 1, j) : rnd_elt<E>(g); break;
            }
            A.at(i, j) = x;
        }
    return A;
}

template<class E>
void arm_e_rung(std::mt19937_64& g, uint32_t k, const std::string& tag) {
    using Tr = r2::Traits<E>;
    const std::vector<std::pair<uint32_t, uint32_t>> shapes = {{0, 7}, {7, 0}, {1, 1}, {1, 200}, {200, 1}, {65, 63}, {129, 257}, {140, 140}};
    for (const auto& [m, n] : shapes)
        for (int kind = 0; kind < 5; ++kind) {
            if (kind == 3 && k >= Tr::kBits) continue;
            const r2::Mat<E> A0 = edge_mat<E>(g, m, n, k, kind);
            const std::string t = tag + " " + std::to_string(m) + "x" + std::to_string(n) + " kind=" + std::to_string(kind);
            r2::Mat<E> Aser = A0;
            r2::EchelonInfo eser;
            r2::SmithResult sser;
            r2::HowellForm<E> hser;
            {
                const sp::ScopedThreads st(1);
                eser = r2::echelon_blocked<Tr>(Aser, k, r2::PivotRule::MinValuation, 64);
                sser = r2::smith<Tr>(A0, k);
                hser = r2::howell<Tr>(A0, k);
            }
            if (kind == 1) check(sser.sigma.empty() && eser.rank == 0, t + ": the zero matrix has rank 0");
            for (uint32_t T : {3u, 16u, 64u})
                for (uint32_t panel : {64u, 128u, 512u}) {   // (PANEL-IMPL) + 512, the large-LU default
                    const sp::ScopedThreads st(T);
                    const sp::ScopedPanel sp_(panel);
                    const std::string tt = t + " T=" + std::to_string(T) + " panel=" + std::to_string(panel);
                    r2::Mat<E> A = A0;
                    const r2::EchelonInfo e = r2::echelon_blocked<Tr>(A, k, r2::PivotRule::MinValuation, panel);
                    check(same_info(e, eser) && same_mat(A, Aser), tt + ": MinValuation LU == serial panel 64, entrywise");
                    const r2::SmithResult s = r2::smith<Tr>(A0, k);
                    check(s.sigma == sser.sigma && s.level_ranks == sser.level_ranks, tt + ": smith == serial");
                    const r2::HowellForm<E> h = r2::howell<Tr>(A0, k);
                    check(h.pivots == hser.pivots && h.pivot_v == hser.pivot_v && same_mat(h.H, hser.H), tt + ": howell == serial, entrywise");
                }
        }
}

void arm_e() {
    std::mt19937_64 g(0x5be0cd19137e2179ull);
    const sp::Stats s0 = sp::stats();
    arm_e_rung<uint32_t>(g, 30, "(E) u32 k=30");
    arm_e_rung<uint32_t>(g, 5, "(E) u32 k=5");
    arm_e_rung<uint64_t>(g, 61, "(E) u64 k=61");
    arm_e_rung<uint64_t>(g, 7, "(E) u64 k=7");
    arm_e_rung<uint8_t>(g, 3, "(E) u8 k=3");
    // ring3k: the same shapes, k = 19 and 2 (k << the 19-trit limit), panels 64 / 128, T = 3, 16, 64
    for (uint32_t k : {19u, 2u})
        for (const auto& [m, n] : std::vector<std::pair<uint32_t, uint32_t>>{{0, 7}, {7, 0}, {1, 1}, {1, 200}, {200, 1}, {65, 63}, {129, 257}})
            for (int kind = 0; kind < 4; ++kind) {
                engine::DenseBuf<uint32_t> A0 = kind < 3 ? rnd3(g, m, n, k, kind) : engine::DenseBuf<uint32_t>(m, n);
                if (kind == 3) {   // all zero; then a high-valuation copy (multiples of 3^(k-1)) on the larger shapes
                    uint32_t q = 1;
                    for (uint32_t i = 0; i + 1 < k; ++i) q *= 3u;
                    if (m * n > 1000)
                        for (uint32_t i = 0; i < m; ++i)
                            for (uint32_t j = 0; j < n; ++j) A0.at(i, j) = static_cast<uint32_t>((g() % 3) * q);
                }
                const std::string t = "(E) ring3k k=" + std::to_string(k) + " " + std::to_string(m) + "x" + std::to_string(n) + " kind=" +
                                      std::to_string(kind);
                r3::SmithResult sser;
                r3::HowellForm hser;
                {
                    const sp::ScopedThreads st(1);
                    sser = r3::smith_levels(A0, k);
                    hser = r3::howell_blocked(A0, k);
                }
                for (uint32_t T : {3u, 16u, 64u})
                    for (uint32_t panel : {64u, 128u}) {
                        const sp::ScopedThreads st(T);
                        const sp::ScopedPanel sp_(panel);
                        const std::string tt = t + " T=" + std::to_string(T) + " panel=" + std::to_string(panel);
                        const r3::SmithResult s = r3::smith_levels(A0, k);
                        check(s.sigma == sser.sigma && s.level_ranks == sser.level_ranks, tt + ": smith_levels == serial");
                        const r3::HowellForm h = r3::howell_blocked(A0, k);
                        check(h.pivots == hser.pivots && h.pivot_v == hser.pivot_v && same_buf(h.H, hser.H), tt + ": howell_blocked == serial");
                    }
            }
    const sp::Stats s1 = sp::stats();
    check(s1.max_threads >= 64 && s1.regions_parallel > s0.regions_parallel, "(E) non-vacuous: a 64-thread region ran");
    std::cout << "  (E) edge shapes 0x7 7x0 1x1 1x200 200x1 65x63 129x257 140x140; all-zero, high-valuation ties, garbage above bit k (k << W: "
                 "u32 k=5, u64 k=7, u8 k=3), duplicate rows; ring3k k = 19, 2; T = 3, 16, 64 x panel 64, 128 (ring2k also 512) == serial (panel 64) entrywise"
              << std::endl;
}

// (S) the shared pool: many regions, each task exactly once; a throwing region then a clean one; nested regions (serial);
// several outside threads calling the threaded Smith at once (one holds the pool, the others run serially): same values.
void arm_s() {
    std::vector<uint32_t> hits(97);
    bool once = true;
    for (uint32_t rep = 0; rep < 2000 && once; ++rep) {
        std::fill(hits.begin(), hits.end(), 0u);
        sp::parallel_for_n(rep % 2 ? 64u : 5u, static_cast<uint32_t>(hits.size()), [&](uint32_t t) { ++hits[t]; });
        for (uint32_t h : hits) once = once && h == 1;
    }
    check(once, "(S) 2000 regions at 5 / 64 threads: every task ran exactly once");
    bool threw = false;
    try {
        sp::parallel_for_n(16, 64, [&](uint32_t t) {
            if (t == 17) throw std::runtime_error("arm_s");
        });
    } catch (const std::runtime_error&) {
        threw = true;
    }
    check(threw, "(S) a task's exception reaches the caller");
    std::fill(hits.begin(), hits.end(), 0u);
    sp::parallel_for_n(16, static_cast<uint32_t>(hits.size()), [&](uint32_t t) { ++hits[t]; });
    check(std::all_of(hits.begin(), hits.end(), [](uint32_t h) { return h == 1; }), "(S) a clean region after a throwing one");
    std::vector<uint32_t> inner(16 * 8, 0);
    sp::parallel_for_n(16, 16, [&](uint32_t t) { sp::parallel_for_n(16, 8, [&](uint32_t u) { ++inner[t * 8 + u]; }); });
    check(std::all_of(inner.begin(), inner.end(), [](uint32_t h) { return h == 1; }), "(S) nested regions (the inner one serial)");
    // concurrent outside callers
    std::mt19937_64 g(99);
    const r2::Mat<uint32_t> A = rnd_mat<uint32_t>(g, 160, 150, 1);
    r2::SmithResult ref;
    {
        const sp::ScopedThreads st(1);
        ref = r2::smith<r2::Traits<uint32_t>>(A, 30);
    }
    std::vector<r2::SmithResult> got(6);
    {
        const sp::ScopedThreads st(16);
        std::vector<std::thread> th;
        for (uint32_t i = 0; i < got.size(); ++i)
            th.emplace_back([&, i] {
                const sp::ScopedEnable en;
                got[i] = r2::smith<r2::Traits<uint32_t>>(A, 30);
            });
        for (auto& x : th) x.join();
    }
    bool same = true;
    for (const auto& s : got) same = same && s.sigma == ref.sigma && s.level_ranks == ref.level_ranks;
    check(same, "(S) six concurrent callers of the threaded smith == serial");
    // (PANEL-IMPL) the size rule of the threaded ring2k panel, and an explicit width winning at every size
    if (sp::explicit_panel() == 0)
        check(sp::descent_panel_2k(sp::kLargePanelRows - 1) == sp::kDefaultPanel && sp::descent_panel_2k(sp::kLargePanelRows) == sp::kLargePanel &&
                  sp::descent_panel() == sp::kDefaultPanel,
              "(S) auto panel: 128 below 12000 rows, 512 from 12000 (ring2k); 128 (ring3k)");
    {
        const sp::ScopedPanel sp_(64);
        check(sp::descent_panel_2k(1) == 64 && sp::descent_panel_2k(1u << 20) == 64 && sp::descent_panel() == 64,
              "(S) an explicit panel wins at every size");
    }
    std::cout << "  (S) the shared pool: 2000 regions, exceptions, nesting, six concurrent callers; default_threads()=" << sp::default_threads()
              << " (" << sp::default_threads_source() << ") panel2k=" << sp::descent_panel_2k(0) << "|"
              << sp::descent_panel_2k(sp::kLargePanelRows) << "@" << sp::kLargePanelRows << " panel3k=" << sp::descent_panel() << std::endl;
}

// ---- the lever -------------------------------------------------------------------------------------------------------
// ---- (PAR-F2) the lazy F_2 / F_3 echelon == the serial one, the whole matrix ------------------------------------------
namespace bin = engine::binary;

// kinds: 0 dense 1/2, 1 sparse 1/64, 2 low rank (product), 3 zero, 4 bottom rows only (long searches),
//        5 wide zero column bands (the window rule), 6 duplicated rows, 7 identity-like with noise
bin::BitMatrix f2_mat(std::mt19937_64& g, uint32_t m, uint32_t n, int kind) {
    bin::BitMatrix B(m, n);
    auto coin = [&](uint32_t d) { return (g() % d) == 0; };
    switch (kind) {
        case 0: for (uint32_t i = 0; i < m; ++i) for (uint32_t j = 0; j < n; ++j) if (coin(2)) B.set(i, j, true); break;
        case 1: for (uint32_t i = 0; i < m; ++i) for (uint32_t j = 0; j < n; ++j) if (coin(64)) B.set(i, j, true); break;
        case 2: {
            const uint32_t rk = 1 + static_cast<uint32_t>(g() % (std::min(m, n) / 3 + 1));
            bin::BitMatrix L(m, rk), R(rk, n);
            for (uint32_t i = 0; i < m; ++i) for (uint32_t j = 0; j < rk; ++j) if (coin(2)) L.set(i, j, true);
            for (uint32_t i = 0; i < rk; ++i) for (uint32_t j = 0; j < n; ++j) if (coin(2)) R.set(i, j, true);
            B = bin::multiply_m4rm(L, R);
            break;
        }
        case 3: break;
        case 4: for (uint32_t i = m - std::min<uint32_t>(m, 5); i < m; ++i) for (uint32_t j = 0; j < n; ++j) if (coin(2)) B.set(i, j, true); break;
        case 8:   // late hits: rows < 1100 zero, below them a quarter of the rows dense (split searches with hits in many tiles)
            for (uint32_t i = std::min<uint32_t>(m, 1100); i < m; ++i)
                if (coin(4)) for (uint32_t j = 0; j < n; ++j) if (coin(2)) B.set(i, j, true);
            break;
        case 5:
            for (uint32_t i = 0; i < m; ++i)
                for (uint32_t j = 0; j < n; ++j)
                    if (((j / 37) % 23 == 0 || (j % 700) < 9) && coin(2)) B.set(i, j, true);
            break;
        case 6: {
            for (uint32_t i = 0; i < m; ++i) {
                if (i > 0 && coin(3)) { const uint32_t s = static_cast<uint32_t>(g() % i); for (uint32_t j = 0; j < n; ++j) B.set(i, j, B.get(s, j)); continue; }
                for (uint32_t j = 0; j < n; ++j) if (coin(3)) B.set(i, j, true);
            }
            break;
        }
        default:
            for (uint32_t i = 0; i < m && i < n; ++i) B.set(i, i, true);
            for (uint32_t t = 0; t < (m + n); ++t) B.set(static_cast<uint32_t>(g() % m), static_cast<uint32_t>(g() % n), true);
            break;
    }
    return B;
}

bin::TritMatrix f3_mat(std::mt19937_64& g, uint32_t m, uint32_t n, int kind) {
    bin::TritMatrix T(m, n);
    auto coin = [&](uint32_t d) { return (g() % d) == 0; };
    auto v12 = [&]() { return 1u + static_cast<uint32_t>(g() & 1u); };
    switch (kind) {
        case 0: for (uint32_t i = 0; i < m; ++i) for (uint32_t j = 0; j < n; ++j) T.set(i, j, static_cast<uint32_t>(g() % 3)); break;
        case 1: for (uint32_t i = 0; i < m; ++i) for (uint32_t j = 0; j < n; ++j) if (coin(64)) T.set(i, j, v12()); break;
        case 2: {
            const uint32_t rk = 1 + static_cast<uint32_t>(g() % (std::min(m, n) / 3 + 1));
            bin::TritMatrix L(m, rk), R(rk, n);
            for (uint32_t i = 0; i < m; ++i) for (uint32_t j = 0; j < rk; ++j) L.set(i, j, static_cast<uint32_t>(g() % 3));
            for (uint32_t i = 0; i < rk; ++i) for (uint32_t j = 0; j < n; ++j) R.set(i, j, static_cast<uint32_t>(g() % 3));
            T = bin::multiply(L, R);
            break;
        }
        case 3: break;
        case 4: for (uint32_t i = m - std::min<uint32_t>(m, 5); i < m; ++i) for (uint32_t j = 0; j < n; ++j) T.set(i, j, static_cast<uint32_t>(g() % 3)); break;
        case 8:
            for (uint32_t i = std::min<uint32_t>(m, 1100); i < m; ++i)
                if (coin(4)) for (uint32_t j = 0; j < n; ++j) T.set(i, j, static_cast<uint32_t>(g() % 3));
            break;
        case 5:
            for (uint32_t i = 0; i < m; ++i)
                for (uint32_t j = 0; j < n; ++j)
                    if (((j / 29) % 17 == 0 || (j % 400) < 5) && coin(2)) T.set(i, j, v12());
            break;
        case 6: {
            for (uint32_t i = 0; i < m; ++i) {
                if (i > 0 && coin(3)) { const uint32_t s = static_cast<uint32_t>(g() % i); for (uint32_t j = 0; j < n; ++j) T.set(i, j, (3u - T.get(s, j)) % 3u); continue; }
                for (uint32_t j = 0; j < n; ++j) if (coin(3)) T.set(i, j, v12());
            }
            break;
        }
        default:
            for (uint32_t i = 0; i < m && i < n; ++i) T.set(i, i, v12());
            for (uint32_t t = 0; t < (m + n); ++t) T.set(static_cast<uint32_t>(g() % m), static_cast<uint32_t>(g() % n), v12());
            break;
    }
    return T;
}

void arm_f() {
    const std::vector<uint32_t> fthreads = {1, 2, 7, 16, 64};
    const std::vector<std::pair<uint32_t, uint32_t>> shapes = {{1, 1}, {1, 300}, {300, 1}, {63, 65}, {129, 513}, {600, 200},
                                                               {200, 1500}, {2100, 700}, {700, 2100}, {3000, 1100},
                                                               {300, 9000}};
    // A moderate threshold (not 0): searches split from ~1 000 rows on, the subset tables from 8 192 columns on (the wide
    // shape), the flushes from ~2^15 words; at 0 every short search would open a region (~10^5 regions, 25 s).
    const sp::ScopedMinWork mwf(uint64_t(1) << 16);
    std::vector<bin::RowPath> paths = {bin::RowPath::Portable};
    for (bin::RowPath p : {bin::RowPath::Avx2, bin::RowPath::Avx512})
        if (bin::row_path_available(p)) paths.push_back(p);
    const sp::Stats s0 = sp::stats();
    std::string names;
    uint32_t cases = 0;
    for (bin::RowPath path : paths) {
        bin::set_row_path(path);
        names += std::string(names.empty() ? "" : ",") + bin::row_path_name(path);
        std::mt19937_64 g(0xF2F3u + static_cast<uint32_t>(path));
        for (const auto& [m, n] : shapes) {
            for (int kind = 0; kind < 9; ++kind) {
                const bool big = static_cast<uint64_t>(m) * n > 1000000ull;
                if (big && path != paths.back() && kind != 1 && kind != 5 && kind != 8) continue;   // the large shapes: three kinds on the forced-off paths
                const std::string t = std::string(bin::row_path_name(path)) + " " + std::to_string(m) + "x" + std::to_string(n) + " kind " +
                                      std::to_string(kind);
                ++cases;
                // F_2
                const bin::BitMatrix B0 = f2_mat(g, m, n, kind);
                bin::BitMatrix Bs = B0;
                bin::BitRref rs;
                {
                    const sp::ScopedThreads st(1);
                    rs = bin::echelon(Bs);
                }
                if (static_cast<uint64_t>(m) * n <= 200000ull) {
                    bin::BitMatrix Bn = B0;
                    check(bin::rref_naive(Bn).pivots == rs.pivots, t + ": F_2 echelon pivots == rref_naive");
                }
                // the large shapes: T = 2, 16, 64 on the native path, T = 7 on the forced-off paths (cost; the small ones at all five)
                const std::vector<uint32_t> tl = !big ? fthreads : (path == paths.back() ? std::vector<uint32_t>{2, 16, 64}
                                                                                          : std::vector<uint32_t>{7});
                for (uint32_t T : tl) {
                    const sp::ScopedThreads st(T);
                    bin::BitMatrix Bl = B0;
                    const bin::BitRref rl = bin::echelon_lazy(Bl);
                    check(rl.rank == rs.rank && rl.pivots == rs.pivots && Bl == Bs,
                          t + " T=" + std::to_string(T) + ": F_2 echelon_lazy == echelon (rank, pivots, whole matrix)");
                }
                // F_3
                const bin::TritMatrix C0 = f3_mat(g, m, n, kind);
                bin::TritMatrix Cs = C0;
                bin::TritRref qs;
                {
                    const sp::ScopedThreads st(1);
                    qs = bin::echelon(Cs);
                }
                if (static_cast<uint64_t>(m) * n <= 200000ull) {
                    bin::TritMatrix Cn = C0;
                    check(bin::rref_naive(Cn).pivots == qs.pivots, t + ": F_3 echelon pivots == rref_naive");
                }
                for (uint32_t T : tl) {
                    const sp::ScopedThreads st(T);
                    bin::TritMatrix Cl = C0;
                    const bin::TritRref ql = bin::echelon_lazy(Cl);
                    check(ql.rank == qs.rank && ql.pivots == qs.pivots && Cl == Cs,
                          t + " T=" + std::to_string(T) + ": F_3 echelon_lazy == echelon (rank, pivots, whole matrix)");
                }
            }
        }
        // pack_low_bits == the per-entry low bits (lengths across the 64 / 512 boundaries)
        for (uint32_t len : {1u, 7u, 63u, 64u, 65u, 127u, 200u, 511u, 513u, 1000u}) {
            std::vector<uint32_t> a(len);
            std::vector<uint64_t> b(len);
            for (uint32_t j = 0; j < len; ++j) { a[j] = static_cast<uint32_t>(g()); b[j] = g(); }
            std::vector<uint64_t> wa((len + 63) / 64, ~uint64_t(0)), wb((len + 63) / 64, ~uint64_t(0)), ref((len + 63) / 64, 0), refb((len + 63) / 64, 0);
            for (uint32_t j = 0; j < len; ++j) {
                ref[j >> 6] |= static_cast<uint64_t>(a[j] & 1u) << (j & 63);
                refb[j >> 6] |= static_cast<uint64_t>(b[j] & 1u) << (j & 63);
            }
            bin::pack_low_bits(a.data(), len, wa.data());
            bin::pack_low_bits(b.data(), len, wb.data());
            check(wa == ref && wb == refb, std::string(bin::row_path_name(path)) + " len " + std::to_string(len) + ": pack_low_bits == low bits");
        }
    }
    bin::set_row_path(bin::RowPath::Auto);
    // compact_rows (the threaded Schur compaction) == the serial ascending memmove: small r0 (buffered waves, buffer
    // smaller than the matrix), large r0 (direct waves), r0 = 0, one row, one column
    {
        std::mt19937_64 g(0xC0u);
        const std::vector<std::array<uint32_t, 3>> cs = {{3000, 3000, 1}, {3000, 3000, 7}, {2000, 5000, 1500}, {4000, 1000, 999},
                                                         {300, 4000, 0}, {1, 9000, 1}, {9000, 1, 1}, {70000, 300, 2}};
        for (const auto& c : cs) {
            const uint32_t m = c[0], n0 = c[1], r0 = c[2];
            if (r0 > m || r0 > n0) continue;
            const uint32_t sr = m - r0, sc = n0 - r0;
            std::vector<uint32_t> a(static_cast<size_t>(m) * n0);
            for (auto& x : a) x = static_cast<uint32_t>(g());
            std::vector<uint32_t> b = a;
            for (uint32_t i = 0; i < sr; ++i) {
                const size_t src0 = static_cast<size_t>(r0 + i) * n0, dst0 = static_cast<size_t>(i) * sc;
                if (src0 != dst0) std::memmove(a.data() + dst0, a.data() + src0, sizeof(uint32_t) * sc);
            }
            for (uint32_t T : {2u, 16u, 64u}) {
                std::vector<uint32_t> bb = b;
                const sp::ScopedThreads st(T);
                sp::compact_rows(bb.data(), r0, n0, sr, sc);
                check(std::equal(a.begin(), a.begin() + static_cast<std::ptrdiff_t>(static_cast<size_t>(sr) * sc), bb.begin()),
                      "(F) compact_rows " + std::to_string(m) + "x" + std::to_string(n0) + " r0=" + std::to_string(r0) + " T=" +
                          std::to_string(T) + " == the serial memmove");
            }
        }
    }
    const sp::Stats s1 = sp::stats();
    check(s1.f2_lazy_calls > s0.f2_lazy_calls && s1.f2_par_searches > s0.f2_par_searches && s1.max_threads >= 64,
          "(F) non-vacuous: lazy echelons ran, pivot searches split, a 64-thread region ran");
    std::cout << "  (F) F_2 / F_3 echelon_lazy == echelon (whole matrix, rank, pivots) at T = 1, 2, 7, 16, 64 on " << cases
              << " adversarial cases (dense, sparse, low rank, zero, bottom rows, wide zero bands, duplicates, noisy identity, late hits); row paths "
              << names << "; lazy calls " << (s1.f2_lazy_calls - s0.f2_lazy_calls) << ", split searches "
              << (s1.f2_par_searches - s0.f2_par_searches) << std::endl;
}

void arm_l() {
    check(sp::threads() == 1, "(L) outside a ScopedEnable: threads() == 1 (every other consumer stays serial)");
    std::mt19937_64 g(7);
    const r2::Mat<uint32_t> A = rnd_mat<uint32_t>(g, 200, 180, 1);
    r2::SmithResult on, off;
    {
        const sp::ScopedLever lv_on(true);   // independent of a set HECKE_DESCENT_SMITH_MT
        const sp::ScopedMinWork mw(0);
        const sp::ScopedEnable en;
        const sp::ScopedThreads st(16);
        check(sp::threads() == 16, "(L) inside a ScopedEnable at ScopedThreads(16): threads() == 16");
        on = r2::smith<r2::Traits<uint32_t>>(A, 30);
        const sp::ScopedLever lv(false);
        check(sp::threads() == 1, "(L) the lever off (scoped): threads() == 1");
        const sp::Stats s0 = sp::stats();
        off = r2::smith<r2::Traits<uint32_t>>(A, 30);
        const sp::Stats s1 = sp::stats();
        check(s1.regions_parallel == s0.regions_parallel && s1.regions_serial == s0.regions_serial,
              "(L) the lever off: no region at all (the serial code verbatim)");
    }
    check(on.sigma == off.sigma && on.level_ranks == off.level_ranks, "(L) lever on == lever off");
    {
        // (PAR-F2) HECKE_DESCENT_SMITH_F2_MT: off = the PAR-2K serial F_2 / F_3 echelon (no lazy call), on == off
        const sp::ScopedLever lv_on(true);
        const sp::ScopedMinWork mw(0);
        const sp::ScopedEnable en;
        const sp::ScopedThreads st(16);
        r2::SmithResult f_on, f_off;
        r3::SmithResult g_on, g_off;
        engine::DenseBuf<uint32_t> A3 = rnd3(g, 220, 190, 19, 1);
        {
            const sp::ScopedF2Lever f(true);
            check(sp::f2_route(), "(L) F2 lever on at 16 threads: f2_route()");
            f_on = r2::smith<r2::Traits<uint32_t>>(A, 30);
            g_on = r3::smith_levels(A3, 19);
        }
        {
            const sp::ScopedF2Lever f(false);
            check(!sp::f2_route(), "(L) F2 lever off: !f2_route()");
            const sp::Stats s0 = sp::stats();
            f_off = r2::smith<r2::Traits<uint32_t>>(A, 30);
            g_off = r3::smith_levels(A3, 19);
            check(sp::stats().f2_lazy_calls == s0.f2_lazy_calls, "(L) F2 lever off: no lazy echelon ran");
        }
        check(f_on.sigma == f_off.sigma && f_on.level_ranks == f_off.level_ranks && f_on.sigma == on.sigma, "(L) F2 lever on == off (ring2k)");
        check(g_on.sigma == g_off.sigma && g_on.level_ranks == g_off.level_ranks, "(L) F2 lever on == off (ring3k)");
    }
    std::cout << "  (L) lever: off = serial verbatim, on = off; " << sp::effective_text() << std::endl;
}

// ---- the read-only measurement hooks ---------------------------------------------------------------------------------
std::vector<uint32_t> parse_u32_list(const char* s) {
    std::vector<uint32_t> v;
    std::stringstream ss(s);
    std::string item;
    while (std::getline(ss, item, ','))
        if (!item.empty()) v.push_back(static_cast<uint32_t>(std::strtoul(item.c_str(), nullptr, 10)));
    return v;
}
double median(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    return v.empty() ? 0.0 : v[v.size() / 2];
}

void hooks() {
    const char* tenv = std::getenv("HECKE_DESCENT_SMITH_MT_THREADS");
    const std::vector<uint32_t> threads = tenv ? parse_u32_list(tenv) : std::vector<uint32_t>{1, 2, 4, 8, 16, 32};
    const char* renv = std::getenv("HECKE_DESCENT_SMITH_MT_REPEATS");
    const uint32_t reps = renv ? std::max<uint32_t>(1, static_cast<uint32_t>(std::strtoul(renv, nullptr, 10))) : 1u;
    if (const char* b = std::getenv("HECKE_DESCENT_SMITH_MT_BENCH")) {
        const auto Ds = parse_u32_list(b);
        std::cout << "EFFECTIVE DescentSmithMtValidation arm=HECKE_DESCENT_SMITH_MT_BENCH cells=" << Ds.size() << " threads=" << threads.size()
                  << " repeats=" << reps << std::endl;
        for (uint32_t D : Ds) {
            std::mt19937_64 g(D);
            // a product of two random factors (inner D/2 + D/8: rank-deficient) plus a 2-power scaled random matrix: several levels
            r2::Mat<uint32_t> A2(D, D);
            {
                const uint32_t r = D / 2 + D / 8;
                r2::Mat<uint32_t> B = rnd_mat<uint32_t>(g, D, r, 0), C = rnd_mat<uint32_t>(g, r, D, 0);
                const sp::ScopedEnable en;
                const sp::ScopedThreads st(0);
                // A2 = B C via the ring2k GEMM kernel (C -= L U on -B), then + 4 * random
                r2::Mat<uint32_t> nb = B;
                for (auto& x : nb.a) x = 0u - x;
                r2::detail::gemm_sub_mt<r2::Traits<uint32_t>>(sp::threads(), A2.a.data(), D, D, nb.a.data(), r, C.a.data(), D, r, D);
                for (auto& x : A2.a) x += 4u * static_cast<uint32_t>(g());
            }
            engine::DenseBuf<uint32_t> A3(D, D);
            {
                const uint32_t M = 1162261467u;
                for (uint32_t i = 0; i < D; ++i)
                    for (uint32_t j = 0; j < D; ++j) A3.at(i, j) = static_cast<uint32_t>(((A2.at(i, j) % M) * (i % 7 == 0 ? 3ull : 1ull)) % M);
            }
            // (tuning only) HECKE_DESCENT_SMITH_MT_PANEL=<w>: the LU panel width of both rungs for this hook, on the serial route
            // (a Params value) and on the threaded route (smithpar::ScopedPanel; PAR-UNIFY): it never changes a value -- the pivots
            // are functions of the exact column values -- and is restored after
            const r2::Params p2saved = r2::params();
            const r3::Params p3saved = r3::params();
            uint32_t hook_panel = 0;
            if (const char* pw = std::getenv("HECKE_DESCENT_SMITH_MT_PANEL")) {
                r2::Params p2 = p2saved;
                r3::Params p3 = p3saved;
                hook_panel = std::max<uint32_t>(1, static_cast<uint32_t>(std::strtoul(pw, nullptr, 10)));
                p2.panel = p3.panel = hook_panel;
                r2::set_params(p2);
                r3::set_params(p3);
            }
            const sp::ScopedPanel hook_sp(hook_panel);
            std::vector<uint32_t> ref2, ref3;
            for (uint32_t T : threads) {
                std::vector<double> t2, t3;
                std::vector<uint32_t> s2, s3;
                uint32_t route_panel = 0;
                for (uint32_t rep = 0; rep < reps; ++rep) {
                    const sp::ScopedEnable en;
                    const sp::ScopedThreads st(T);
                    route_panel = sp::route_panel(r2::params().panel);
                    auto t = clk::now();
                    s2 = r2::smith<r2::Traits<uint32_t>>(A2, 30).sigma;
                    t2.push_back(since(t));
                    t = clk::now();
                    s3 = r3::smith(A3, 19).sigma;
                    t3.push_back(since(t));
                }
                if (ref2.empty()) { ref2 = s2; ref3 = s3; }
                check(s2 == ref2 && s3 == ref3, "(hook BENCH) D=" + std::to_string(D) + " T=" + std::to_string(T) + ": sigma == the first thread count's");
                std::cout << "  PAR2K_BENCH D=" << D << " threads=" << T << " t_ring2k=" << fx(median(t2)) << " t_ring3k=" << fx(median(t3))
                          << " rank2=" << s2.size() << " rank3=" << s3.size() << " reps=" << reps << " panel=" << route_panel
                          << std::endl;
            }
            r2::set_params(p2saved);
            r3::set_params(p3saved);
        }
    }
    if (const char* c = std::getenv("HECKE_DESCENT_SMITH_MT_CELLS")) {
        std::vector<std::pair<uint32_t, uint32_t>> cells;
        std::stringstream ss(c);
        std::string item;
        while (std::getline(ss, item, ',')) {
            const size_t p = item.find(':');
            if (p == std::string::npos) continue;
            cells.emplace_back(static_cast<uint32_t>(std::strtoul(item.substr(0, p).c_str(), nullptr, 10)),
                               static_cast<uint32_t>(std::strtoul(item.substr(p + 1).c_str(), nullptr, 10)));
        }
        std::cout << "EFFECTIVE DescentSmithMtValidation arm=HECKE_DESCENT_SMITH_MT_CELLS cells=" << cells.size() << " threads=" << threads.size()
                  << " repeats=" << reps << std::endl;
        for (const auto& [N, n] : cells) {
            const auto h = betti(N, n);
            const auto cc = integral_descent::LevelNClosedForm(N, n);
            for (uint32_t ell : {2u, 3u}) {
                integral_descent::LocalPiece ref;
                bool have = false;
                for (uint32_t T : threads) {
                    std::vector<double> ts, tw;
                    integral_descent::LocalPiece P;
                    for (uint32_t rep = 0; rep < reps; ++rep) {
                        const sp::ScopedThreads st(T);
                        const auto t = clk::now();
                        P = integral_descent::RunTreeLocalPieceN(ell, N, n, ell == 2 ? 4u : 6u, h);
                        tw.push_back(since(t));
                        ts.push_back(P.smith_seconds);
                    }
                    if (!have) { ref = P; have = true; }
                    check(same_piece(P, ref), "(hook CELLS) (" + std::to_string(N) + "," + std::to_string(n) + ") ell=" + std::to_string(ell) +
                                                  " T=" + std::to_string(T) + ": == the first thread count's");
                    std::cout << "  PAR2K_CELL N=" << N << " n=" << n << " D=" << cc.D << " ell=" << ell << " threads=" << T
                              << " t_piece=" << fx(median(tw)) << " t_smith=" << fx(median(ts)) << " certified=" << P.all_certified()
                              << " reps=" << reps << std::endl;
                }
            }
        }
    }
}

} // namespace

uint32_t DescentSmithMtValidation() {
    std::cout << "=== DescentSmithMtValidation (S24 PAR-2K: the threaded ring2k / ring3k Smith == serial, bit for bit) ===" << std::endl;
    const auto t0 = clk::now();
    g_checks = g_fail = 0;
    {
        const sp::ScopedEnable en;           // the arms call ring2k / ring3k directly: opt in as the descent does
        const sp::ScopedMinWork mw(0);       // every step splits, however small
        const sp::ScopedLever lv(true);      // the threaded route even when HECKE_DESCENT_SMITH_MT=0 is set
        const sp::Stats a = sp::stats();
        arm_r2();
        const sp::Stats b = sp::stats();
        arm_r3();
        const sp::Stats c = sp::stats();
        check(b.panel_splits > a.panel_splits && b.regions_parallel > a.regions_parallel,
              "(R2) non-vacuous: split panel updates (fused pivot search) and parallel regions ran");
        check(c.panel_splits > b.panel_splits && c.regions_parallel > b.regions_parallel,
              "(R3) non-vacuous: split panel updates (fused pivot search) and parallel regions ran");
        arm_e();
        arm_s();
        {
            const sp::ScopedF2Lever f2(true);   // (PAR-F2) independent of a set HECKE_DESCENT_SMITH_F2_MT
            arm_f();
        }
        std::cout << "  (R2)/(R3) split panel updates " << (b.panel_splits - a.panel_splits) << " / " << (c.panel_splits - b.panel_splits)
                  << ", parallel regions " << (b.regions_parallel - a.regions_parallel) << " / " << (c.regions_parallel - b.regions_parallel)
                  << std::endl;
    }
    {
        const sp::ScopedMinWork mw(0);
        const sp::ScopedLever lv(true);
        arm_d();
    }
    arm_l();
    hooks();
    std::cout << "DescentSmithMtValidation: " << g_checks << " checks, " << g_fail << " failing, " << fx(since(t0)) << " s" << std::endl;
    return g_fail;
}
