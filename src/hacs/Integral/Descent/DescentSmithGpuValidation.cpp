// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// (slot-24 PAR-GPU) DescentSmithGpuValidation: the device route of the ring2k LU against the host code.  See the header.

#include "DescentSmithGpuValidation.h"

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
#include <string>
#include <vector>

#include "DescentAssembly.h"
#include "DescentLevelN.h"
#include "DescentSmithGpu.h"
#include "DescentTree.h"
#include "../../../ModArithmetic/Binary/Ring2k.h"
#include "../../../ModArithmetic/Binary/Ring2kDevice.h"
#include "../../../ModArithmetic/Binary/SmithPar.h"
#include "../../../Cuda/CudaCtx.h"
#include "../../../Cuda/CudaRing2k.h"
#include "../../../Engine/EngineRuntime.h"

namespace {

namespace r2 = engine::ring2k;
namespace dv = engine::ring2k::device;
namespace sp = engine::smithpar;
namespace sg = integral_descent::smith_gpu;
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

template<class E>
E rnd_elt(std::mt19937_64& g) {
    return static_cast<E>(g());
}

// Adversarial inputs.  0 uniform; 1 a product B C of inner dimension 2/3 min(m, n) (rank-deficient over Z, several levels);
// 2 rows scaled by 2^(i mod 5); 3 sparse 2-powers (most entries 0, the rest 2^v u); 4 odd entries only in the bottom rows
// (every pivot is a row swap); 5 all entries even except a sparse diagonal-ish set.
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
        return A;
    }
    for (uint32_t i = 0; i < m; ++i)
        for (uint32_t j = 0; j < n; ++j) {
            E x = rnd_elt<E>(g);
            switch (kind) {
            case 2: x = Tr::shl(x, i % 5); break;
            case 3: x = (g() % 5 == 0) ? Tr::shl(E(x | 1u), static_cast<uint32_t>(g() % 6)) : E(0); break;
            case 4: if (i < m / 2) x = Tr::shl(x, 1); break;
            case 5: x = ((i + 3 * j) % 7 == 0) ? E(x | 1u) : Tr::shl(x, 1 + static_cast<uint32_t>(g() % 3)); break;
            default: break;
            }
            A.at(i, j) = x;
        }
    return A;
}

template<class E>
bool same_mat(const r2::Mat<E>& a, const r2::Mat<E>& b) {
    return a.rows == b.rows && a.cols == b.cols && a.a == b.a;
}
bool same_info(const r2::EchelonInfo& a, const r2::EchelonInfo& b) {
    return a.rank == b.rank && a.pivots == b.pivots && a.pivot_v == b.pivot_v;
}

// The F_2 rank profile moved to the front (as smith() does before its FirstOdd LU); returns r0.
template<class E>
uint32_t odd_prepare(r2::Mat<E>& A) {
    using Tr = r2::Traits<E>;
    engine::binary::BitMatrix B(A.rows, A.cols);
    for (uint32_t i = 0; i < A.rows; ++i)
        for (uint32_t j = 0; j < A.cols; ++j) if (Tr::odd(A.at(i, j))) B.set(i, j, true);
    const engine::binary::BitRref br = engine::binary::echelon(B);
    std::vector<uint32_t> order(br.pivots.begin(), br.pivots.end());
    for (uint32_t j = 0, q = 0; j < A.cols; ++j) {
        if (q < br.pivots.size() && br.pivots[q] == j) { ++q; continue; }
        order.push_back(j);
    }
    const r2::Mat<E> A0 = A;
    for (uint32_t i = 0; i < A.rows; ++i)
        for (uint32_t j = 0; j < A.cols; ++j) A.at(i, j) = A0.at(i, order[j]);
    return br.rank;
}

// One echelon_blocked comparison: host (device off) against the forced device route; the device must really have run.
template<class E>
void cmp_lu(const r2::Mat<E>& A0, uint32_t k, r2::PivotRule rule, uint32_t climit, uint32_t panel, uint32_t T, uint32_t max_cols,
            const std::string& tag, uint64_t store_bytes = 0) {
    using Tr = r2::Traits<E>;
    if (climit == 0) return;
    r2::Mat<E> Ah = A0, Ad = A0;
    r2::EchelonInfo eh, ed;
    {
        const sg::ScopedForce off(false);
        const sp::ScopedThreads st(T);
        eh = r2::echelon_blocked<Tr>(Ah, k, rule, panel, climit);
    }
    const dv::Stats s0 = dv::stats();
    {
        const sg::ScopedForce on(true, max_cols, false, store_bytes);
        const sp::ScopedThreads st(T);
        ed = r2::echelon_blocked<Tr>(Ad, k, rule, panel, climit);
    }
    const dv::Stats s1 = dv::stats();
    const bool ran = s1.device == s0.device + 1;
    const bool want_tiled = max_cols > 0 && max_cols < A0.cols;
    check(ran, tag + ": the device LU ran");
    if (want_tiled) check(s1.tiled == s0.tiled + 1 && s1.slabs >= s0.slabs + 2, tag + ": the tiled route ran (>= 2 slabs)");
    check(same_info(eh, ed) && same_mat(Ah, Ad), tag + ": device LU == host LU, entrywise (matrix, rank, pivots, valuations)");
}

template<class E>
void arm_e_rung(std::mt19937_64& g, const std::vector<uint32_t>& ks, const std::string& rung) {
    const std::vector<std::pair<uint32_t, uint32_t>> shapes = {{37, 41}, {150, 130}, {261, 300}, {300, 170}, {1, 77}, {90, 1}, {130, 129}};
    for (uint32_t k : ks)
        for (const auto& [m, n] : shapes)
            for (int kind = 0; kind < 6; ++kind) {
                const r2::Mat<E> A0 = rnd_mat<E>(g, m, n, kind);
                const std::string t = "(E) " + rung + " k=" + std::to_string(k) + " " + std::to_string(m) + "x" + std::to_string(n) +
                                      " kind=" + std::to_string(kind);
                const bool big = m >= 150 && n >= 130;
                // MinValuation on the raw matrix; FirstOdd after the F_2 profile, col_limit = r0
                r2::Mat<E> Aodd = A0;
                const uint32_t r0 = odd_prepare(Aodd);
                for (uint32_t panel : {64u, 16u, 128u, 1u}) {
                    if (panel == 1 && !(m <= 150 && n <= 130)) continue;
                    for (uint32_t T : {1u, 16u}) {
                        if (panel != 64 && T == 16 && kind % 2) continue;   // keep the arm short: the T x panel grid on half the kinds
                        const std::string tt = t + " panel=" + std::to_string(panel) + " T=" + std::to_string(T);
                        cmp_lu<E>(A0, k, r2::PivotRule::MinValuation, n, panel, T, 0, tt + " MinValuation resident");
                        cmp_lu<E>(Aodd, k, r2::PivotRule::FirstOdd, r0, panel, T, 0, tt + " FirstOdd resident");
                        if (big && panel != 1) {
                            const uint32_t mc = panel == 64 ? 128u : panel == 16 ? 16u : 128u;   // 2 .. 19 slabs
                            cmp_lu<E>(A0, k, r2::PivotRule::MinValuation, n, panel, T, mc, tt + " MinValuation tiled cw=" + std::to_string(mc));
                            cmp_lu<E>(Aodd, k, r2::PivotRule::FirstOdd, r0, panel, T, mc, tt + " FirstOdd tiled cw=" + std::to_string(mc));
                            // the L store capped at one byte: every phase holds one panel (the most phases, replays, permutations)
                            cmp_lu<E>(A0, k, r2::PivotRule::MinValuation, n, panel, T, mc, tt + " MinValuation tiled one-panel phases", 1);
                            cmp_lu<E>(Aodd, k, r2::PivotRule::FirstOdd, r0, panel, T, mc, tt + " FirstOdd tiled one-panel phases", 1);
                        }
                    }
                }
            }
    // the zero matrix: no pivot at all under MinValuation
    const r2::Mat<E> Z(70, 90);
    cmp_lu<E>(Z, ks.front(), r2::PivotRule::MinValuation, 90, 64, 1, 0, "(E) " + rung + " zero 70x90");
}

void arm_e() {
    std::mt19937_64 g(0x5a17c0de2bull);
    arm_e_rung<uint32_t>(g, {30, 32, 7}, "u32");
    arm_e_rung<uint64_t>(g, {61, 64, 13}, "u64");
    std::cout << "  (E) echelon_blocked: device == host entrywise; u32 k = 30 / 32 / 7, u64 k = 61 / 64 / 13; six input kinds; panels "
                 "64 / 16 / 128 / 1; T = 1, 16; resident and tiled windows"
              << std::endl;
}

template<class E>
void arm_k_rung(std::mt19937_64& g, const std::string& rung) {
    using Tr = r2::Traits<E>;
    const std::vector<std::array<uint32_t, 3>> shapes = {{1, 1, 1},   {1, 130, 5},  {130, 1, 64},  {63, 65, 16},  {64, 64, 64},
                                                         {65, 129, 65}, {200, 300, 130}, {17, 33, 0}, {129, 64, 17}, {3, 200, 128}};
    for (const auto& sh : shapes) {
        const uint32_t m = sh[0], n = sh[1], k = sh[2];
        std::vector<E> C(static_cast<size_t>(m) * n), L(static_cast<size_t>(m) * std::max<uint32_t>(k, 1)), U(static_cast<size_t>(std::max<uint32_t>(k, 1)) * n);
        for (auto& x : C) x = rnd_elt<E>(g);
        for (auto& x : L) x = rnd_elt<E>(g);
        for (auto& x : U) x = rnd_elt<E>(g);
        std::vector<E> Ch = C, Cd = C;
        if (k > 0) Tr::gemm_sub(Ch.data(), n, m, L.data(), k, U.data(), n, k, n);
        engine::cuda::ring2k::gemm_sub_host(sizeof(E), Cd.data(), n, L.data(), k, U.data(), n, m, n, k);
        check(Ch == Cd, "(K) " + rung + " " + std::to_string(m) + "x" + std::to_string(n) + " K=" + std::to_string(k) +
                            ": device GEMM == host GEMM, entrywise");
    }
}

void arm_k() {
    std::mt19937_64 g(0x6b2e9f11ull);
    arm_k_rung<uint32_t>(g, "u32");
    arm_k_rung<uint64_t>(g, "u64");
    std::cout << "  (K) the device GEMM C -= L U == the host kernel on 10 edge shapes, u32 and u64" << std::endl;
}

void arm_s() {
    std::mt19937_64 g(0x77aa11ull);
    for (int kind = 0; kind < 6; ++kind)
        for (const auto& [m, n] : std::vector<std::pair<uint32_t, uint32_t>>{{180, 210}, {260, 140}}) {
            const std::string t = "(S) " + std::to_string(m) + "x" + std::to_string(n) + " kind=" + std::to_string(kind);
            const r2::Mat<uint32_t> A = rnd_mat<uint32_t>(g, m, n, kind);
            const r2::Mat<uint64_t> B = rnd_mat<uint64_t>(g, m, n, kind);
            r2::SmithResult sh32, sh64, sd32, sd64;
            r2::HowellForm<uint32_t> hh32, hd32;
            r2::HowellForm<uint64_t> hh64, hd64;
            {
                const sg::ScopedForce off(false);
                sh32 = r2::smith<r2::Traits<uint32_t>>(A, 30);
                sh64 = r2::smith<r2::Traits<uint64_t>>(B, 61);
                hh32 = r2::howell<r2::Traits<uint32_t>>(A, 32);
                hh64 = r2::howell<r2::Traits<uint64_t>>(B, 40);
            }
            {
                const sg::ScopedForce on(true);
                sd32 = r2::smith<r2::Traits<uint32_t>>(A, 30);
                sd64 = r2::smith<r2::Traits<uint64_t>>(B, 61);
                hd32 = r2::howell<r2::Traits<uint32_t>>(A, 32);
                hd64 = r2::howell<r2::Traits<uint64_t>>(B, 40);
            }
            check(sh32.sigma == sd32.sigma && sh32.level_ranks == sd32.level_ranks, t + " u32 k=30: smith device == host");
            check(sh64.sigma == sd64.sigma && sh64.level_ranks == sd64.level_ranks, t + " u64 k=61: smith device == host");
            check(hh32.pivots == hd32.pivots && hh32.pivot_v == hd32.pivot_v && same_mat(hh32.H, hd32.H), t + " u32 k=32: howell device == host");
            check(hh64.pivots == hd64.pivots && hh64.pivot_v == hd64.pivot_v && same_mat(hh64.H, hd64.H), t + " u64 k=40: howell device == host");
        }
    std::cout << "  (S) smith (sigma, level_ranks) and howell (H, pivots) through the device LU == host, u32 and u64" << std::endl;
}

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
    const dv::Stats s0 = dv::stats();
    uint32_t cells = 0;
    for (const auto& [N, n] : std::vector<std::pair<uint32_t, uint32_t>>{{11, 6}, {13, 10}, {25, 4}, {9, 8}, {7, 20}}) {
        const auto h = betti(N, n);
        integral_descent::LocalPiece host, devr, devs, devp;
        {
            const sg::ScopedForce off(false);
            host = integral_descent::RunTreeLocalPieceN(2, N, n, 4, h);
        }
        {
            const sg::ScopedForce on(true);
            devr = integral_descent::RunTreeLocalPieceN(2, N, n, 4, h);
        }
        {
            const sg::ScopedForce on(true, 128);
            devs = integral_descent::RunTreeLocalPieceN(2, N, n, 4, h);
        }
        {
            const sg::ScopedForce on(true, 128, false, 1);
            devp = integral_descent::RunTreeLocalPieceN(2, N, n, 4, h);
        }
        const std::string t = "(D) tree ell=2 (" + std::to_string(N) + "," + std::to_string(n) + ")";
        check(same_piece(host, devr), t + " resident: every differential and the torsion == host");
        check(same_piece(host, devs), t + " tiled cw=128: every differential and the torsion == host");
        check(same_piece(host, devp), t + " tiled cw=128, one-panel phases: every differential and the torsion == host");
        ++cells;
    }
    for (bool tree : {true, false}) {
        const integral_descent::ScopedDescentTree sdt(tree);
        std::vector<integral_model::AbelianGroupType> Hh, Hd;
        bool rh = true, rd = true;
        {
            const sg::ScopedForce off(false);
            const auto R = integral_descent::AssembleDescentN(11, 6);
            rh = R.refused;
            Hh = R.H;
        }
        {
            const sg::ScopedForce on(true);
            const auto R = integral_descent::AssembleDescentN(11, 6);
            rd = R.refused;
            Hd = R.H;
        }
        bool same = !rh && !rd && Hh.size() == Hd.size();
        for (size_t q = 0; same && q < Hh.size(); ++q) same = Hh[q] == Hd[q];
        check(same, std::string("(D) AssembleDescentN(11, 6) ") + (tree ? "tree" : "dense") + ": H^0..H^5 device == host");
    }
    const dv::Stats s1 = dv::stats();
    check(s1.device > s0.device && s1.tiled > s0.tiled, "(D) the real inputs reached the device (resident and tiled)");
    std::cout << "  (D) real inputs: RunTreeLocalPieceN ell = 2 on " << cells << " cells (resident and tiled) and AssembleDescentN(11, 6) "
              << "tree + dense == host; device LUs " << (s1.device - s0.device) << std::endl;
}

void arm_v() {
    std::mt19937_64 g(99);
    const r2::Mat<uint64_t> A = rnd_mat<uint64_t>(g, 160, 150, 1);
    const dv::Stats s0 = dv::stats();
    r2::SmithResult sd, sh;
    {
        const sg::ScopedForce on(true, 0, true);
        sd = r2::smith<r2::Traits<uint64_t>>(A, 61);
    }
    {
        const sg::ScopedForce off(false);
        sh = r2::smith<r2::Traits<uint64_t>>(A, 61);
    }
    const dv::Stats s1 = dv::stats();
    check(s1.verified > s0.verified && s1.verified - s0.verified == s1.device - s0.device,
          "(V) the verify twin re-ran every device LU on the host (and found it equal)");
    check(sd.sigma == sh.sigma, "(V) smith under the verify twin == host");
    std::cout << "  (V) verify twin: " << (s1.verified - s0.verified) << " device LUs re-run on the host, equal" << std::endl;
}

void arm_f() {
    std::mt19937_64 g(5);
    const r2::Mat<uint32_t> A = rnd_mat<uint32_t>(g, 140, 150, 1);
    r2::Mat<uint32_t> Ah = A, Ad = A;
    r2::EchelonInfo eh, ed;
    {
        const sg::ScopedForce off(false);
        eh = r2::echelon_blocked<r2::Traits<uint32_t>>(Ah, 30, r2::PivotRule::MinValuation, 64);
    }
    const dv::Stats s0 = dv::stats();
    {
        const sg::ScopedForce on(true);
        const sg::ScopedSimulateError err;
        ed = r2::echelon_blocked<r2::Traits<uint32_t>>(Ad, 30, r2::PivotRule::MinValuation, 64);
    }
    const dv::Stats s1 = dv::stats();
    check(s1.fallbacks == s0.fallbacks + 1 && s1.device == s0.device, "(F) a CudaError at open: one fallback, no device LU");
    check(same_info(eh, ed) && same_mat(Ah, Ad), "(F) the fallback's host LU == host");
    std::cout << "  (F) simulated CudaError at session open: the host LU ran, same values" << std::endl;
}

void arm_p() {
    const dv::Ops* o = dv::ops();
    check(sg::installed() && o != nullptr && o->plan != nullptr, "(P) the device table is installed");
    if (!o || !o->plan) return;
    {
        const engine::tape::ScopedThreadDevice cpu(engine::tape::ThreadDevice::Cpu);
        check(o->plan(4, 5000, 5000, 64) == 0 && o->plan(8, 8000, 8000, 64) == 0,
              "(P) policy cpu (thread-local), no override: plan() declines, the host LU runs");
    }
    {
        const engine::tape::ScopedThreadDevice gpu(engine::tape::ThreadDevice::Gpu);
        const uint32_t cw_small = o->plan(4, 1000, 1000, 64);
        if (!sg::lever_on() || sg::min_entries() > 1000000ull)
            check(cw_small == 0, "(P) policy gpu below the size floor (1000 x 1000 < ENGINE_GPU_SMITH2K_MIN) or the lever off: plan() declines");
        else   // ENGINE_GPU_SMITH2K_MIN lowered to <= 10^6 (read-only tuning lever): the small matrix plans resident
            check(cw_small == 1000, "(P) policy gpu, ENGINE_GPU_SMITH2K_MIN <= 10^6: plan() takes 1000 x 1000 resident");
        const uint32_t cw = o->plan(4, 5000, 5000, 64);
        if (sg::lever_on() && sg::min_entries() <= 25000000ull)
            check(cw == 5000, "(P) policy gpu, 5000 x 5000 u32 (100 MB): plan() takes the device, resident");
        else
            check(cw == 0, "(P) ENGINE_GPU_SMITH2K=0 or a raised floor: plan() declines");
    }
    std::cout << "  (P) policy: cpu declines, gpu below the floor declines, gpu above it plans resident; " << sg::effective_text() << std::endl;
}

std::vector<std::pair<uint32_t, uint32_t>> parse_cells(const char* c) {
    std::vector<std::pair<uint32_t, uint32_t>> cells;
    std::stringstream ss(c);
    std::string item;
    while (std::getline(ss, item, ',')) {
        const size_t p = item.find(':');
        if (p == std::string::npos) continue;
        cells.emplace_back(static_cast<uint32_t>(std::strtoul(item.substr(0, p).c_str(), nullptr, 10)),
                           static_cast<uint32_t>(std::strtoul(item.substr(p + 1).c_str(), nullptr, 10)));
    }
    return cells;
}

void hooks() {
    const char* c = std::getenv("HECKE_DESCENT_SMITH_GPU_CELLS");
    if (!c) return;
    const auto cells = parse_cells(c);
    const char* arms = std::getenv("HECKE_DESCENT_SMITH_GPU_ARMS");
    const bool gpu_only = arms && std::strcmp(arms, "gpu") == 0;
    std::cout << "EFFECTIVE DescentSmithGpuValidation arm=HECKE_DESCENT_SMITH_GPU_CELLS cells=" << cells.size() << " arms=" << (gpu_only ? "gpu" : "cpu,gpu")
              << " threads=" << sp::threads() << "(outside scope) ENGINE_THREADS=" << (std::getenv("ENGINE_THREADS") ? std::getenv("ENGINE_THREADS") : "-")
              << std::endl;
    for (const auto& [N, n] : cells) {
        const auto h = betti(N, n);
        const auto cc = integral_descent::LevelNClosedForm(N, n);
        integral_descent::LocalPiece ref;
        bool have = false;
        for (int arm = gpu_only ? 1 : 0; arm < 2; ++arm) {
            const dv::Stats s0 = dv::stats();
            const sg::ScopedForce f(arm == 1);
            const auto t = clk::now();
            const auto P = integral_descent::RunTreeLocalPieceN(2, N, n, 4, h);
            const double tw = since(t);
            const dv::Stats s1 = dv::stats();
            if (have) check(same_piece(P, ref), "(hook CELLS) (" + std::to_string(N) + "," + std::to_string(n) + ") ell=2: gpu == cpu");
            if (!have) { ref = P; have = true; }
            std::string nu;
            for (const auto& d : P.d) {
                uint64_t sum = 0;
                for (uint32_t a : d.nonunit) sum += a;
                nu += "[" + std::to_string(d.rows) + "x" + std::to_string(d.cols) + " rK=" + std::to_string(d.r_K) + " W=" + std::to_string(d.W) +
                      " nonunit=" + std::to_string(d.nonunit.size()) + "/" + std::to_string(sum) + "]";
            }
            auto sec = [](uint64_t ns) { return fx(static_cast<double>(ns) * 1e-9, 2); };
            std::cout << "  PARGPU_CELL N=" << N << " n=" << n << " D=" << cc.D << " ell=2 arm=" << (arm ? "gpu" : "cpu") << " t_piece=" << fx(tw)
                      << " t_smith=" << fx(P.smith_seconds) << " certified=" << P.all_certified() << " device_lus=" << (s1.device - s0.device)
                       << " tiled=" << (s1.tiled - s0.tiled) << " slabs=" << (s1.slabs - s0.slabs) << " panels=" << (s1.panels - s0.panels) << " dev_s=open:" << sec(s1.ns_open - s0.ns_open)
                      << ",fetch:" << sec(s1.ns_fetch - s0.ns_fetch) << ",panel:" << sec(s1.ns_panel - s0.ns_panel) << ",step:" << sec(s1.ns_step - s0.ns_step)
                      << ",host:" << sec(s1.ns_host - s0.ns_host) << ",close:" << sec(s1.ns_close - s0.ns_close) << " d=" << nu << std::endl;
        }
    }
}

} // namespace

uint32_t DescentSmithGpuValidation() {
    if (!engine::cuda::available()) {
        std::cout << "[SKIP] DescentSmithGpuValidation: no usable CUDA device (" << engine::cuda::unavailable_reason()
                  << "): the device Smith LU is never taken; the host LU is the only route" << std::endl;
        return 0;
    }
    std::cout << "=== DescentSmithGpuValidation (S24 PAR-GPU: the device-resident 2-adic Smith LU == host, bit for bit) ===" << std::endl;
    const auto t0 = clk::now();
    g_checks = g_fail = 0;
    {
        const sp::ScopedEnable en;        // as the descent: the host side may thread (T = 16 arms)
        const sp::ScopedLever lv(true);
        const sp::ScopedMinWork mw(0);
        arm_k();
        arm_e();
        arm_s();
        arm_v();
        arm_f();
    }
    arm_d();
    arm_p();
    const dv::Stats s = dv::stats();
    check(s.device > 0 && s.tiled > 0 && s.replays > 0 && s.lookaheads > 0 && s.panels > 0,
          "non-vacuous: device LUs, tiled LUs, panel replays and lookahead panels ran");
    std::cout << "  counters: device=" << s.device << " resident=" << s.resident << " tiled=" << s.tiled << " slabs=" << s.slabs
              << " replays=" << s.replays << " panels=" << s.panels
              << " lookahead=" << s.lookaheads << " fallbacks=" << s.fallbacks << " verified=" << s.verified << std::endl;
    hooks();
    std::cout << "DescentSmithGpuValidation: " << g_checks << " checks, " << g_fail << " failing, " << fx(since(t0)) << " s" << std::endl;
    return g_fail;
}
