// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// Ring2kLadderValidation (slot-22 BIN-2).  See Ring2kLadderValidation.h and ENGINE_API.md Sec. 8.2.
//
// Parts:
//   (A) the rung table (select_rung / next_rung / rungs_for);
//   (B) wraparound fuzz: every rung's arithmetic vs nf::BigInt mod 2^W (u8 exhaustively), Newton
//       inverses, valuations, shifts, masks, word round trips; every kernel path's AXPY / GEMM vs a
//       scalar loop (tails included);
//   (C) random and planted matrices at k across every rung: Smith valuations, Howell form, kernel
//       module, length on EVERY rung able to carry k, vs ModCtx::prime_power(2, k) (u32 for k <= 31,
//       u64 for k <= 63), vs engine::ladic (k >= 64, and spot checks below), vs R6 (nf::BigInt), vs the
//       unblocked references; structure of the Howell form, A K^T = 0, rank-nullity, the level-0 rank
//       = the F_2 rank (BitMatrix), planted Smith forms recovered, panel / block sweeps, kernel paths;
//   (D) amalgam matrices: the 7-vertex model's A_Cq over Z/2^k (hecke::GenerateSL2ZRelationsRing),
//       and the exact Sym^n coboundary [rho(S) - 1; rho(U) - 1] and cocycle relation matrix
//       [[N_S, 0], [0, N_U], [1 + S, -(1 + U + U^2)]] of SL_2(Z) = Z/4 *_{Z/2} Z/6 (nf::BigInt,
//       rho_n cross-checked against hecke::RhoOfExactRing), reduced to many k: every rung vs the
//       engine / ladic / R6, and the cross-k consistency of the Smith valuations;
//   (E) the adaptive rung (smith_adaptive) on planted invariants 2^70, 2^150 and on a near-cap one;
//   (F) negative controls: a carry-dropping R5 product and a one-step-short Newton inverse on R3 are
//       caught by the comparisons; a corrupted output entry is caught;
//   (G) timing (RING2K_TIMING / RING2K_ENGINE_TIMING lines; reported, never gated).

#include "Ring2kLadderValidation.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <functional>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "../../ModArithmetic/Binary/BitMatrix.h"
#include "../../ModArithmetic/Binary/Ring2k.h"
#include "Ring2kReference.h"
#include "../../ModArithmetic/DenseBuf.h"
#include "../../ModArithmetic/DenseElim.h"
#include "../../ModArithmetic/DenseView.h"
#include "../../ModArithmetic/LadicInduction.h"
#include "../../ModArithmetic/ModCtx.h"
#include "../Hecke/CoefficientRing.h"
#include "../Hecke/ExactInt.h"
#include "../LocalSystemSL2Z.h"
#include "../LocalSystemSimplicialComplex.h"
#include "../NumberField/BigInt.h"

namespace {

namespace r2 = engine::ring2k;
namespace ref6 = engine::ring2k::ref6;
namespace lad = engine::ladic;
namespace elim = engine::elim;
namespace binary = engine::binary;
using engine::DenseBuf;
using engine::cview;
using engine::view;
using r2::WideMatrix;
using r2::WideHowell;
using r2::RungChoice;
using nf::BigInt;
using u128 = unsigned __int128;

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
double now_s() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

std::string sig_str(const std::vector<uint32_t>& s) {
    std::string o = "[";
    for (size_t i = 0; i < s.size() && i < 12; ++i) o += (i ? "," : "") + str(s[i]);
    if (s.size() > 12) o += ",...(" + str(s.size()) + ")";
    return o + "]";
}

// ---------------------------------------------------------------------------
// Independent multiword helpers (generators and the A K^T check only).
// ---------------------------------------------------------------------------
void w_mul_add(uint64_t* acc, const uint64_t* a, const uint64_t* b, uint32_t nw) {   // acc += a b mod 2^(64 nw)
    std::vector<uint64_t> p(nw, 0);
    for (uint32_t i = 0; i < nw; ++i) {
        if (!a[i]) continue;
        uint64_t c = 0;
        for (uint32_t j = 0; i + j < nw; ++j) {
            const u128 t = static_cast<u128>(a[i]) * b[j] + p[i + j] + c;
            p[i + j] = static_cast<uint64_t>(t);
            c = static_cast<uint64_t>(t >> 64);
        }
    }
    uint64_t c = 0;
    for (uint32_t i = 0; i < nw; ++i) {
        const u128 t = static_cast<u128>(acc[i]) + p[i] + c;
        acc[i] = static_cast<uint64_t>(t);
        c = static_cast<uint64_t>(t >> 64);
    }
}
bool w_zero_mod(const uint64_t* a, uint32_t nw, uint32_t k) {
    for (uint32_t i = 0; i < nw; ++i) {
        uint64_t x = a[i];
        if (64 * (i + 1) > k) x &= (64 * i >= k) ? 0 : ((uint64_t(1) << (k - 64 * i)) - 1);
        if (x) return false;
    }
    return true;
}

// Random families.
enum class Fam { Uniform, Even, Small, Sparse, LowRank };
const char* fam_name(Fam f) {
    switch (f) {
    case Fam::Uniform: return "uniform";
    case Fam::Even: return "even";
    case Fam::Small: return "small";
    case Fam::Sparse: return "sparse";
    case Fam::LowRank: return "lowrank";
    }
    return "?";
}
WideMatrix random_wide(uint32_t m, uint32_t n, uint32_t k, Fam f, uint64_t& s) {
    WideMatrix A(m, n, k);
    const uint32_t nw = A.nw;
    if (f == Fam::LowRank) {
        const uint32_t r = 1 + static_cast<uint32_t>(splitmix(s) % (std::max<uint32_t>(1, std::min(m, n) / 2)));
        WideMatrix L = random_wide(m, r, k, Fam::Uniform, s), R = random_wide(r, n, k, Fam::Small, s);
        for (uint32_t i = 0; i < m; ++i)
            for (uint32_t j = 0; j < n; ++j)
                for (uint32_t t = 0; t < r; ++t) w_mul_add(A.entry(i, j), L.entry(i, t), R.entry(t, j), nw);
        A.canonicalise();
        return A;
    }
    const uint32_t c = 1 + static_cast<uint32_t>(splitmix(s) % 3);
    for (uint32_t i = 0; i < m; ++i)
        for (uint32_t j = 0; j < n; ++j) {
            uint64_t* e = A.entry(i, j);
            switch (f) {
            case Fam::Uniform:
                for (uint32_t t = 0; t < nw; ++t) e[t] = splitmix(s);
                break;
            case Fam::Even: {   // 2^c * uniform
                std::vector<uint64_t> u(nw);
                for (uint32_t t = 0; t < nw; ++t) u[t] = splitmix(s);
                for (uint32_t t = nw; t-- > 0;) e[t] = (u[t] << c) | (t ? u[t - 1] >> (64 - c) : 0);
                break;
            }
            case Fam::Small: {   // in [-3, 3]
                const int64_t v = static_cast<int64_t>(splitmix(s) % 7) - 3;
                for (uint32_t t = 0; t < nw; ++t) e[t] = v < 0 ? ~uint64_t(0) : 0;
                e[0] = static_cast<uint64_t>(v);
                break;
            }
            case Fam::Sparse:
                if (splitmix(s) % 8 == 0)
                    for (uint32_t t = 0; t < nw; ++t) e[t] = splitmix(s) >> (splitmix(s) % 4 == 0 ? 0 : 40);
                break;
            case Fam::LowRank: break;
            }
        }
    A.canonicalise();
    return A;
}

// Planted P_r L D R P_c over Z/2^(64 nw): L unit lower, R unit upper, D = diag(2^e_i u_i) (u_i odd) or 0.
WideMatrix planted(uint32_t n, uint32_t k, const std::vector<uint32_t>& e, uint64_t& s) {
    WideMatrix L(n, n, k), R(n, n, k), D(n, n, k), A(n, n, k), T(n, n, k);
    const uint32_t nw = A.nw;
    for (uint32_t i = 0; i < n; ++i)
        for (uint32_t j = 0; j < n; ++j) {
            if (i == j) { L.entry(i, j)[0] = 1; R.entry(i, j)[0] = 1; continue; }
            WideMatrix& M = j < i ? L : R;
            if (splitmix(s) % 3 == 0) continue;
            for (uint32_t t = 0; t < nw; ++t) M.entry(i, j)[t] = splitmix(s);
        }
    for (uint32_t i = 0; i < n; ++i) {
        if (e[i] >= 64 * nw) continue;   // a zero diagonal entry
        std::vector<uint64_t> u(nw);
        for (uint32_t t = 0; t < nw; ++t) u[t] = splitmix(s);
        u[0] |= 1u;
        const uint32_t q = e[i] / 64, b = e[i] % 64;
        uint64_t* d = D.entry(i, i);
        for (uint32_t t = q; t < nw; ++t) d[t] = (u[t - q] << b) | (b && t > q ? u[t - q - 1] >> (64 - b) : 0);
    }
    for (uint32_t i = 0; i < n; ++i)
        for (uint32_t j = 0; j < n; ++j)
            w_mul_add(T.entry(i, j), L.entry(i, j), D.entry(j, j), nw);   // T = L D
    for (uint32_t i = 0; i < n; ++i)
        for (uint32_t j = 0; j < n; ++j)
            for (uint32_t t = 0; t < n; ++t) w_mul_add(A.entry(i, j), T.entry(i, t), R.entry(t, j), nw);
    std::vector<uint32_t> pr(n), pc(n);
    for (uint32_t i = 0; i < n; ++i) pr[i] = pc[i] = i;
    for (uint32_t i = n; i-- > 1;) { std::swap(pr[i], pr[splitmix(s) % (i + 1)]); std::swap(pc[i], pc[splitmix(s) % (i + 1)]); }
    WideMatrix P(n, n, k);
    for (uint32_t i = 0; i < n; ++i)
        for (uint32_t j = 0; j < n; ++j)
            for (uint32_t t = 0; t < nw; ++t) P.entry(pr[i], pc[j])[t] = A.entry(i, j)[t];
    P.canonicalise();
    return P;
}

// ---------------------------------------------------------------------------
// Engine adapters.
// ---------------------------------------------------------------------------
template<class Elt>
DenseBuf<Elt> to_engine(const WideMatrix& A) {
    DenseBuf<Elt> B(A.rows, A.cols);
    for (uint32_t i = 0; i < A.rows; ++i)
        for (uint32_t j = 0; j < A.cols; ++j) B.at(i, j) = static_cast<Elt>(A.entry(i, j)[0]);
    return B;
}
template<class Elt>
WideHowell from_engine(const elim::HowellResult<Elt>& h, uint32_t n, uint32_t k) {
    WideHowell w;
    w.H = WideMatrix(h.rows(), n, k);
    for (uint32_t i = 0; i < h.rows(); ++i)
        for (uint32_t j = 0; j < n; ++j) w.H.entry(i, j)[0] = static_cast<uint64_t>(h.H.at(i, j));
    w.pivots = h.pivots;
    w.pivot_v = h.pivot_v;
    return w;
}
std::vector<uint32_t> sigma_to_invariant_factors(const r2::SmithResult& s) {
    std::vector<uint32_t> e;
    for (uint32_t x : s.sigma) e.push_back(s.k - x);
    std::sort(e.rbegin(), e.rend());
    return e;
}

lad::LadicMat<uint64_t> to_ladic(const WideMatrix& A) {
    const lad::DigitLayout<uint64_t> lay = lad::DigitLayout<uint64_t>::make(2, A.k);
    lad::LadicMat<uint64_t> M(lay, A.rows, A.cols);
    std::vector<uint32_t> bits(A.k);
    for (uint32_t i = 0; i < A.rows; ++i)
        for (uint32_t j = 0; j < A.cols; ++j) {
            const uint64_t* e = A.entry(i, j);
            for (uint32_t b = 0; b < A.k; ++b) bits[b] = (e[b / 64] >> (b % 64)) & 1u;
            M.set_ell_digits(i, j, bits);
        }
    return M;
}
WideHowell from_ladic(const lad::Howell<uint64_t>& h, uint32_t n, uint32_t k) {
    WideHowell w;
    w.H = WideMatrix(h.rows(), n, k);
    for (uint32_t i = 0; i < h.rows(); ++i)
        for (uint32_t j = 0; j < n; ++j) {
            const std::vector<uint32_t> bits = h.H.ell_digits(i, j);
            uint64_t* e = w.H.entry(i, j);
            for (uint32_t b = 0; b < k; ++b) if (bits[b]) e[b / 64] |= uint64_t(1) << (b % 64);
        }
    w.pivots = h.pivots;
    w.pivot_v = h.pivot_v;
    return w;
}

binary::BitMatrix mod2(const WideMatrix& A) {
    binary::BitMatrix B(A.rows, A.cols);
    for (uint32_t i = 0; i < A.rows; ++i)
        for (uint32_t j = 0; j < A.cols; ++j) if (A.entry(i, j)[0] & 1u) B.set(i, j, true);
    return B;
}

// Conditions 1-3 of the Howell form (ENGINE_API.md Sec. 13.5), read off the words.
bool is_howell_shape(const WideHowell& h) {
    const WideMatrix& H = h.H;
    if (h.pivots.size() != H.rows || h.pivot_v.size() != H.rows) return false;
    for (uint32_t i = 0; i < H.rows; ++i) {
        if (i && h.pivots[i] <= h.pivots[i - 1]) return false;
        const uint32_t pc = h.pivots[i], v = h.pivot_v[i];
        if (pc >= H.cols || v >= H.k) return false;
        for (uint32_t j = 0; j < pc; ++j)
            for (uint32_t t = 0; t < H.nw; ++t) if (H.entry(i, j)[t]) return false;
        const uint64_t* p = H.entry(i, pc);
        for (uint32_t t = 0; t < H.nw; ++t) if (p[t] != (t == v / 64 ? uint64_t(1) << (v % 64) : 0)) return false;
        for (uint32_t i2 = 0; i2 < i; ++i2) {   // entry above < 2^v as an integer
            const uint64_t* x = H.entry(i2, pc);
            for (uint32_t b = v; b < 64 * H.nw; ++b) if ((x[b / 64] >> (b % 64)) & 1u) return false;
        }
    }
    return true;
}
// A x == 0 mod 2^k for every row x of K (the right kernel).
bool annihilates(const WideMatrix& A, const WideHowell& K) {
    const uint32_t nw = A.nw;
    std::vector<uint64_t> acc(nw);
    for (uint32_t r = 0; r < K.H.rows; ++r)
        for (uint32_t i = 0; i < A.rows; ++i) {
            std::fill(acc.begin(), acc.end(), 0);
            for (uint32_t j = 0; j < A.cols; ++j) w_mul_add(acc.data(), A.entry(i, j), K.H.entry(r, j), nw);
            if (!w_zero_mod(acc.data(), nw, A.k)) return false;
        }
    return true;
}

// The rungs exercised at k: every non-R5 rung with capacity >= k, the first two R5 word counts.
std::vector<RungChoice> test_rungs(uint32_t k, uint32_t max_r5 = 2) {
    std::vector<RungChoice> v;
    uint32_t r5 = 0;
    for (const RungChoice& c : r2::rungs_for(k)) {
        if (c.rung == r2::Rung::R5 && r5++ >= max_r5) break;
        v.push_back(c);
    }
    return v;
}

struct Opts {
    bool engine = true;     // ModCtx direct (k <= 63) / ladic (k >= 64)
    bool ladic_small_k = false;   // also ladic at k <= 63
    bool engine_u32 = false;
    bool ref6 = false;
    bool naive = false;
    bool kernel = true;
    bool paths = false;
    uint32_t max_r5 = 2;
    const std::vector<uint32_t>* planted = nullptr;
};

struct Tally { uint64_t matrices = 0, rung_runs = 0, engine_cmp = 0, ladic_cmp = 0, ref6_cmp = 0, naive_cmp = 0; } g_t;

r2::SmithResult compare_matrix(const std::string& tag0, const WideMatrix& A, const Opts& o) {
    const uint32_t k = A.k;
    const std::string tag = tag0 + " k=" + str(k) + " " + str(A.rows) + "x" + str(A.cols);
    ++g_t.matrices;
    const std::vector<RungChoice> rungs = test_rungs(k, o.max_r5);
    const RungChoice prim = rungs.front();
    const r2::SmithResult S0 = r2::smith_on(prim, A);
    const WideHowell H0 = r2::howell_on(prim, A);
    WideHowell K0;
    if (o.kernel) K0 = r2::kernel_module_on(prim, A);
    // structure
    check(is_howell_shape(H0), tag + ": Howell conditions 1-3 [" + r2::rung_label(prim) + "]");
    check(S0.length() == H0.length(), tag + ": length(Smith) == length(Howell)");
    check(std::is_sorted(S0.sigma.begin(), S0.sigma.end()) && (S0.sigma.empty() || S0.sigma.back() < k),
          tag + ": sigma ascending, < k");
    {
        const uint32_t r2f = binary::rank(mod2(A));
        const uint32_t lr0 = S0.sigma.empty() ? 0 : static_cast<uint32_t>(std::count(S0.sigma.begin(), S0.sigma.end(), 0u));
        check(lr0 == r2f, tag + ": #{sigma = 0} == the F_2 rank (BitMatrix)");
    }
    if (o.kernel) {
        check(is_howell_shape(K0), tag + ": kernel module is a Howell form");
        check(annihilates(A, K0), tag + ": A * kernel^T == 0 mod 2^k");
        check(K0.length() + S0.length() == A.cols * k, tag + ": length(ker) + length(im) == n k");
    }
    if (o.planted) check(S0.sigma == *o.planted, tag + ": planted Smith valuations recovered (got " + sig_str(S0.sigma) +
                                                     ", want " + sig_str(*o.planted) + ")");
    // every other rung
    for (size_t t = 1; t < rungs.size(); ++t) {
        const RungChoice& c = rungs[t];
        const std::string rt = tag + " [" + r2::rung_label(c) + " vs " + r2::rung_label(prim) + "]";
        check(r2::smith_on(c, A).sigma == S0.sigma, rt + ": Smith valuations");
        check(r2::howell_on(c, A) == H0, rt + ": Howell form entrywise");
        if (o.kernel && A.rows * A.cols <= 4000) check(r2::kernel_module_on(c, A) == K0, rt + ": kernel module entrywise");
        ++g_t.rung_runs;
    }
    // the kernel paths of the native rungs
    if (o.paths) {
        for (r2::KernelPath p : {r2::KernelPath::Portable, r2::KernelPath::Avx2, r2::KernelPath::Avx512}) {
            if (!r2::kernel_path_available(p)) continue;
            r2::set_kernel_path(p);
            for (const RungChoice& c : rungs) {
                if (c.rung == r2::Rung::R0 || c.rung == r2::Rung::R4 || c.rung == r2::Rung::R5) continue;
                const std::string rt = tag + " [" + r2::rung_label(c) + " path " + r2::kernel_path_name(p) + "]";
                check(r2::smith_on(c, A).sigma == S0.sigma, rt + ": Smith valuations");
                check(r2::howell_on(c, A) == H0, rt + ": Howell form");
            }
            r2::set_kernel_path(r2::KernelPath::Auto);
        }
    }
    const bool nonempty = A.rows > 0 && A.cols > 0;
    // the engine
    if (o.engine && nonempty && k >= 2 && k <= 63) {
        const engine::ModCtx<uint64_t> ctx = engine::ModCtx<uint64_t>::prime_power(2, k);
        const DenseBuf<uint64_t> B = to_engine<uint64_t>(A);
        check(from_engine(elim::howell<uint64_t>(ctx, cview(B)), A.cols, k) == H0, tag + ": Howell == engine (u64 ModCtx Z/2^k)");
        check(elim::invariant_factors<uint64_t>(ctx, cview(B)) == sigma_to_invariant_factors(S0),
              tag + ": invariant factors == engine (u64)");
        check(elim::length<uint64_t>(ctx, cview(B)) == S0.length(), tag + ": length == engine (u64)");
        if (o.kernel) check(from_engine(elim::kernel_module<uint64_t>(ctx, cview(B)), A.cols, k) == K0,
                            tag + ": kernel module == engine (u64)");
        if (o.engine_u32 && k <= 31) {
            const engine::ModCtx<uint32_t> c32 = engine::ModCtx<uint32_t>::prime_power(2u, k);
            const DenseBuf<uint32_t> B32 = to_engine<uint32_t>(A);
            check(from_engine(elim::howell<uint32_t>(c32, cview(B32)), A.cols, k) == H0, tag + ": Howell == engine (u32)");
            check(elim::invariant_factors<uint32_t>(c32, cview(B32)) == sigma_to_invariant_factors(S0),
                  tag + ": invariant factors == engine (u32)");
        }
        ++g_t.engine_cmp;
    }
    if (o.engine && nonempty && k >= 2 && (k >= 64 || o.ladic_small_k)) {
        const lad::LadicMat<uint64_t> L = to_ladic(A);
        const lad::Smith<uint64_t> sm = lad::smith_lift(L);
        std::vector<uint32_t> ls = sm.s;
        std::sort(ls.begin(), ls.end());
        check(ls == S0.sigma, tag + ": Smith valuations == engine::ladic (" + sig_str(ls) + " vs " + sig_str(S0.sigma) + ")");
        check(from_ladic(lad::howell(L), A.cols, k) == H0, tag + ": Howell == engine::ladic");
        if (o.kernel) check(from_ladic(lad::kernel_module(L), A.cols, k) == K0, tag + ": kernel module == engine::ladic");
        ++g_t.ladic_cmp;
    }
    if (o.ref6) {
        check(ref6::smith(A).sigma == S0.sigma, tag + ": Smith valuations == R6 (nf::BigInt)");
        check(ref6::howell(A) == H0, tag + ": Howell == R6 (nf::BigInt)");
        ++g_t.ref6_cmp;
    }
    if (o.naive && prim.rung != r2::Rung::R0) {
        check(r2::smith_naive_on(prim, A).sigma == S0.sigma, tag + ": blocked Smith == global-pivot reference");
        check(r2::howell_naive_on(prim, A) == H0, tag + ": blocked Howell == the unblocked Sec. 13.5 pass");
        ++g_t.naive_cmp;
    }
    return S0;
}

// ---------------------------------------------------------------------------
// (A) the rung table
// ---------------------------------------------------------------------------
void part_rungs() {
    const struct { uint32_t k; const char* label; } want[] = {
        {1, "R0(bits)"}, {2, "R1(u8)"}, {8, "R1(u8)"}, {9, "R2(u16)"}, {16, "R2(u16)"}, {17, "R2(u32)"}, {32, "R2(u32)"},
        {33, "R3(u64)"}, {64, "R3(u64)"}, {65, "R4(u128)"}, {128, "R4(u128)"}, {129, "R5(N=3)"}, {192, "R5(N=3)"},
        {193, "R5(N=4)"}, {320, "R5(N=5)"}, {321, "R5(N=6)"}, {1000, "R5(N=16)"}, {1025, "R5(N=20)"}, {4096, "R5(N=64)"}};
    for (const auto& w : want) {
        const RungChoice c = r2::select_rung(w.k);
        check(r2::rung_label(c) == w.label && c.capacity >= w.k, "select_rung(" + str(w.k) + ") = " + r2::rung_label(c));
    }
    bool threw = false;
    try { (void)r2::select_rung(4097); } catch (const std::logic_error&) { threw = true; }
    check(threw, "select_rung(4097) refused (R5 tops out at N = 64)");
    // next_rung walks the whole ladder with strictly growing capacity
    RungChoice c = r2::select_rung(1);
    uint32_t steps = 0, prev = 0;
    std::string walk;
    while (c.capacity) {
        check(c.capacity > prev, "next_rung: capacity grows at " + r2::rung_label(c));
        prev = c.capacity;
        walk += (walk.empty() ? "" : " ") + r2::rung_label(c);
        c = r2::next_rung(c);
        ++steps;
    }
    check(prev == 4096 && steps == 6 + r2::r5_word_counts().size(), "the ladder ends at R5(N=64)");
    std::cout << "  ladder: " << walk << std::endl;
}

// ---------------------------------------------------------------------------
// (B) wraparound fuzz
// ---------------------------------------------------------------------------
BigInt big_of(const uint64_t* w, uint32_t nw) { return ref6::from_words(w, nw); }

template<class E>
void fuzz_type(uint64_t& s, uint32_t pairs) {
    using Tr = r2::Traits<E>;
    const uint32_t W = Tr::kBits, nw = Tr::kWords;
    const BigInt mod = BigInt::pow(BigInt(2), W);
    std::vector<std::vector<uint64_t>> vals;
    auto push = [&](std::vector<uint64_t> v) { v.resize(nw, 0); vals.push_back(v); };
    push({0}); push({1}); push({2}); push({3});
    {   // 2^(W-1), 2^W - 1, 2^(W-1) +- 1, word-boundary bits
        std::vector<uint64_t> top(nw, 0), ones(nw, ~uint64_t(0));
        if (W % 64) { ones[nw - 1] = (uint64_t(1) << (W % 64)) - 1; top[nw - 1] = uint64_t(1) << (W % 64 - 1); }
        else top[nw - 1] = uint64_t(1) << 63;
        push(top); push(ones);
        std::vector<uint64_t> t1 = top; t1[0] |= 1; push(t1);
        for (uint32_t b = 63; b + 1 < W; b += 64) {
            std::vector<uint64_t> x(nw, 0); x[b / 64] = uint64_t(1) << 63; push(x);
            std::vector<uint64_t> y(nw, 0); y[(b + 1) / 64] = 1; push(y);
            std::vector<uint64_t> z(nw, 0); for (uint32_t t = 0; t <= b / 64; ++t) z[t] = ~uint64_t(0); push(z);
        }
    }
    while (vals.size() < 64) {
        std::vector<uint64_t> v(nw);
        for (auto& x : v) x = splitmix(s);
        if (W < 64) v[0] &= (uint64_t(1) << W) - 1;
        push(v);
    }
    auto toE = [&](const std::vector<uint64_t>& v) { return Tr::from_words(v.data(), nw); };
    auto big = [&](const E& x) { std::vector<uint64_t> w(nw); Tr::to_words(x, w.data(), nw); return big_of(w.data(), nw); };
    uint32_t bad = 0, n = 0;
    for (uint32_t t = 0; t < pairs; ++t) {
        const std::vector<uint64_t>& va = vals[t < vals.size() ? t : splitmix(s) % vals.size()];
        const std::vector<uint64_t>& vb = vals[splitmix(s) % vals.size()];
        const E a = toE(va), b = toE(vb);
        const BigInt A = big_of(va.data(), nw), B = big_of(vb.data(), nw);
        ++n;
        if (big(Tr::add(a, b)) != (A + B).floor_mod(mod)) ++bad;
        if (big(Tr::sub(a, b)) != (A - B).floor_mod(mod)) ++bad;
        if (big(Tr::mul(a, b)) != (A * B).floor_mod(mod)) ++bad;
        if (big(Tr::neg(a)) != (-A).floor_mod(mod)) ++bad;
        const uint32_t k = 1 + static_cast<uint32_t>(splitmix(s) % W);
        const BigInt mk = BigInt::pow(BigInt(2), k);
        if (big(Tr::mask(a, k)) != A.floor_mod(mk)) ++bad;
        if (Tr::zero_mod(a, k) != A.floor_mod(mk).is_zero()) ++bad;
        if (Tr::val(a, k) != ref6::valuation(A.floor_mod(mk), k)) ++bad;
        const uint32_t v = static_cast<uint32_t>(splitmix(s) % (W + 1));
        if (big(Tr::shr(a, v)) != A / BigInt::pow(BigInt(2), v)) ++bad;
        if (big(Tr::shl(a, v)) != (A * BigInt::pow(BigInt(2), v)).floor_mod(mod)) ++bad;
        if (Tr::ctz(a) != (A.is_zero() ? W : ref6::valuation(A, W + 1))) ++bad;
        if (Tr::odd(a) != (A.floor_mod(BigInt(2)) == BigInt(1))) ++bad;
        // Newton
        E u = a;
        if (!Tr::odd(u)) u = Tr::add(u, Tr::one());
        const E ui = Tr::inv_odd(u);
        if (!Tr::eq(Tr::mul(u, ui), Tr::one())) ++bad;
        // word round trip
        std::vector<uint64_t> w(nw + 1, 0xdeadbeefULL);
        Tr::to_words(a, w.data(), nw + 1);
        if (w[nw] != 0 || !Tr::eq(Tr::from_words(w.data(), nw + 1), a)) ++bad;
    }
    check(bad == 0, std::string("wraparound fuzz ") + Tr::name() + " (" + str(W) + " bits): " + str(n) + " cases, " + str(bad) +
                        " wrong vs nf::BigInt mod 2^W");
}

void fuzz_u8_exhaustive() {
    using Tr = r2::Traits<uint8_t>;
    uint32_t bad = 0;
    for (uint32_t a = 0; a < 256; ++a) {
        for (uint32_t b = 0; b < 256; ++b) {
            if (Tr::add(uint8_t(a), uint8_t(b)) != uint8_t((a + b) & 255u)) ++bad;
            if (Tr::sub(uint8_t(a), uint8_t(b)) != uint8_t((a - b) & 255u)) ++bad;
            if (Tr::mul(uint8_t(a), uint8_t(b)) != uint8_t((a * b) & 255u)) ++bad;
        }
        if (a & 1u) {
            const uint8_t ui = Tr::inv_odd(uint8_t(a));
            if (((a * ui) & 255u) != 1u) ++bad;
        }
        for (uint32_t k = 1; k <= 8; ++k)
            if (Tr::val(uint8_t(a), k) != std::min<uint32_t>(k, (a & ((1u << k) - 1)) ? __builtin_ctz(a) : k)) ++bad;
    }
    check(bad == 0, "u8 arithmetic exhaustive (65536 pairs, 128 inverses, 2048 valuations): " + str(bad) + " wrong");
}

template<class T>
void fuzz_kernels(uint64_t& s) {
    using Wd = typename std::conditional<(sizeof(T) < sizeof(unsigned)), unsigned, T>::type;
    uint32_t bad = 0, cases = 0;
    for (r2::KernelPath p : {r2::KernelPath::Portable, r2::KernelPath::Avx2, r2::KernelPath::Avx512}) {
        if (!r2::kernel_path_available(p)) continue;
        r2::set_kernel_path(p);
        for (uint32_t t = 0; t < 40; ++t) {
            const size_t n = splitmix(s) % 300;
            std::vector<T> r(n), pv(n), want(n);
            for (size_t j = 0; j < n; ++j) { r[j] = T(splitmix(s)); pv[j] = T(splitmix(s)); }
            const T f = T(splitmix(s));
            for (size_t j = 0; j < n; ++j) want[j] = T(Wd(r[j]) - Wd(f) * Wd(pv[j]));
            r2::kernel_axpy(r.data(), pv.data(), f, n);
            if (r != want) ++bad;
            // GEMM: C (mr x nc) -= L (mr x np) U (np x nc), with leading dimensions > widths
            const uint32_t mr = static_cast<uint32_t>(splitmix(s) % 11), npv = static_cast<uint32_t>(1 + splitmix(s) % 9),
                           nc = static_cast<uint32_t>(splitmix(s) % 270);
            const size_t ldc = nc + 3, ldl = npv + 1, ldu = nc + 5;
            std::vector<T> C(mr * ldc), L(mr * ldl), U(npv * ldu);
            for (auto& x : C) x = T(splitmix(s));
            for (auto& x : L) x = (splitmix(s) % 4 == 0) ? T(0) : T(splitmix(s));
            for (auto& x : U) x = T(splitmix(s));
            std::vector<T> Cw = C;
            for (uint32_t i = 0; i < mr; ++i)
                for (uint32_t q = 0; q < npv; ++q)
                    for (uint32_t j = 0; j < nc; ++j)
                        Cw[i * ldc + j] = T(Wd(Cw[i * ldc + j]) - Wd(L[i * ldl + q]) * Wd(U[q * ldu + j]));
            r2::kernel_gemm_sub(C.data(), ldc, mr, L.data(), ldl, U.data(), ldu, npv, nc);
            if (C != Cw) ++bad;
            cases += 2;
        }
        // BIN-2 audit F6(a): the multi-tile branch of gemm_body (ncols beyond one L2 tile of
        // 2^18 / (sizeof(T) np) columns), three cases per path.
        for (uint32_t t = 0; t < 3; ++t) {
            const uint32_t npv = 64, mr = 5 + t;
            const uint32_t jt = static_cast<uint32_t>((size_t(1) << 18) / (sizeof(T) * npv));
            const uint32_t nc = 2 * jt + 37 + 11 * t;
            const size_t ldc = nc + 1, ldl = npv + 2, ldu = nc + 3;
            std::vector<T> C(mr * ldc), L(mr * ldl), U(npv * ldu);
            for (auto& x : C) x = T(splitmix(s));
            for (auto& x : L) x = (splitmix(s) % 5 == 0) ? T(0) : T(splitmix(s));
            for (auto& x : U) x = T(splitmix(s));
            std::vector<T> Cw = C;
            for (uint32_t i = 0; i < mr; ++i)
                for (uint32_t q = 0; q < npv; ++q)
                    for (uint32_t j = 0; j < nc; ++j)
                        Cw[i * ldc + j] = T(Wd(Cw[i * ldc + j]) - Wd(L[i * ldl + q]) * Wd(U[q * ldu + j]));
            r2::kernel_gemm_sub(C.data(), ldc, mr, L.data(), ldl, U.data(), ldu, npv, nc);
            if (C != Cw) ++bad;
            cases += 1;
        }
    }
    r2::set_kernel_path(r2::KernelPath::Auto);
    check(bad == 0, "kernel AXPY / GEMM on every path, " + str(8 * sizeof(T)) + "-bit: " + str(cases) + " cases, " + str(bad) + " wrong");
}

void part_fuzz() {
    uint64_t s = 0xF022ULL;
    fuzz_u8_exhaustive();
    fuzz_type<uint8_t>(s, 3000);
    fuzz_type<uint16_t>(s, 3000);
    fuzz_type<uint32_t>(s, 3000);
    fuzz_type<uint64_t>(s, 3000);
    fuzz_type<u128>(s, 3000);
    fuzz_type<r2::WordN<3>>(s, 1500);
    fuzz_type<r2::WordN<4>>(s, 1500);
    fuzz_type<r2::WordN<5>>(s, 1000);
    fuzz_type<r2::WordN<8>>(s, 600);
    fuzz_type<r2::WordN<16>>(s, 300);
    fuzz_type<r2::WordN<64>>(s, 60);
    fuzz_kernels<uint8_t>(s);
    fuzz_kernels<uint16_t>(s);
    fuzz_kernels<uint32_t>(s);
    fuzz_kernels<uint64_t>(s);
}

// ---------------------------------------------------------------------------
// (C) random and planted matrices
// ---------------------------------------------------------------------------
uint32_t part_random() {
    uint64_t s = 0xB2A9ULL;
    const uint32_t klist[] = {1, 2, 3, 5, 8, 9, 13, 16, 17, 24, 31, 32, 33, 47, 63, 64, 65, 100, 127, 128, 129, 192, 256, 300, 513, 1024};
    uint32_t count = 0;
    for (uint32_t k : klist) {
        const bool big_k = k > 256;
        // shapes: (rows, cols, family, options)
        struct Case { uint32_t m, n; Fam f; };
        std::vector<Case> cases = {{0, 5, Fam::Uniform}, {4, 0, Fam::Uniform}, {1, 1, Fam::Uniform}, {3, 7, Fam::Small},
                                   {7, 3, Fam::Even},    {9, 9, Fam::Sparse},  {12, 12, Fam::Uniform}, {10, 14, Fam::LowRank},
                                   {14, 11, Fam::Even},  {30, 45, Fam::Small}, {45, 30, Fam::LowRank}, {50, 50, Fam::Even}};
        if (!big_k) {
            cases.push_back({70, 70, Fam::Uniform});
            cases.push_back({90, 130, Fam::Sparse});
            cases.push_back({130, 90, Fam::Even});
        }
        if (k <= 64) cases.push_back({200, 200, Fam::LowRank});
        for (size_t ci = 0; ci < cases.size(); ++ci) {
            const Case& c = cases[ci];
            const WideMatrix A = random_wide(c.m, c.n, k, c.f, s);
            Opts o;
            const uint64_t sz = static_cast<uint64_t>(c.m) * c.n;
            o.ref6 = sz <= 196 && k <= 1024;
            o.naive = sz <= 10000;
            o.engine_u32 = k <= 31 && ci % 3 == 0;
            o.engine = k <= 63 ? sz <= 40000 : sz <= 2500;
            o.ladic_small_k = k >= 2 && k <= 63 && sz <= 400 && ci % 4 == 1;
            o.paths = k <= 64 && ci % 5 == 2;
            o.kernel = sz <= 10000;
            o.max_r5 = big_k ? 1 : 2;
            compare_matrix(std::string("random#") + str(count) + " " + fam_name(c.f), A, o);
            ++count;
        }
        // planted Smith forms (two per k): valuations 0, small, near k, and zeros
        for (uint32_t pi = 0; pi < 2; ++pi) {
            const uint32_t n = pi == 0 ? 10 : (big_k ? 24 : 80);
            const uint32_t nw = (k + 63) / 64;
            std::vector<uint32_t> e(n), want;
            for (uint32_t i = 0; i < n; ++i) {
                const uint64_t z = splitmix(s) % 10;
                e[i] = z < 4 ? 0 : z < 7 ? static_cast<uint32_t>(splitmix(s) % std::min<uint32_t>(k, 6))
                     : z < 9 ? static_cast<uint32_t>(splitmix(s) % k) : 64 * nw;   // last: a zero diagonal entry
                if (e[i] < k) want.push_back(e[i]);
            }
            std::sort(want.begin(), want.end());
            const WideMatrix A = planted(n, k, e, s);
            Opts o;
            o.planted = &want;
            o.ref6 = n <= 10 && k <= 1024;
            o.naive = true;
            o.engine = k <= 63 || n <= 24;
            o.kernel = n <= 30;
            o.max_r5 = big_k ? 1 : 2;
            compare_matrix("planted#" + str(pi) + " n=" + str(n), A, o);
            ++count;
        }
    }
    // BIN-2 audit F6(b): one planted 3 x 3 at the top of the ladder, k = 4096 (R5 N = 64; engine::ladic).
    {
        const std::vector<uint32_t> e = {4095, 0, 2000};
        std::vector<uint32_t> want = {0, 2000, 4095};
        const WideMatrix A = planted(3, 4096, e, s);
        Opts o;
        o.planted = &want;
        o.naive = true;
        o.engine = true;
        o.kernel = true;
        o.max_r5 = 1;
        compare_matrix("planted k=4096 n=3", A, o);
        ++count;
    }
    // Panel / block sweeps: values never depend on the tunables.
    {
        const r2::Params saved = r2::params();
        for (uint32_t k : {7u, 40u, 100u, 300u}) {
            const WideMatrix A = random_wide(150, 170, k, Fam::Even, s);
            const WideMatrix B = random_wide(160, 120, k, Fam::LowRank, s);
            const r2::SmithResult sa = r2::smith_wide(A), sb = r2::smith_wide(B);
            const WideHowell ha = r2::howell_wide(A), hb = r2::howell_wide(B);
            for (uint32_t panel : {1u, 3u, 8u, 33u, 200u})
                for (uint32_t rb : {1u, 5u, 64u, 300u}) {
                    r2::Params p; p.panel = panel; p.reduce_block = rb;
                    r2::set_params(p);
                    const std::string t = "sweep k=" + str(k) + " panel=" + str(panel) + " reduce_block=" + str(rb);
                    check(r2::smith_wide(A).sigma == sa.sigma && r2::smith_wide(B).sigma == sb.sigma, t + ": Smith unchanged");
                    check(r2::howell_wide(A) == ha && r2::howell_wide(B) == hb, t + ": Howell unchanged");
                }
            r2::set_params(saved);
        }
    }
    return count;
}

// ---------------------------------------------------------------------------
// (D) amalgam matrices
// ---------------------------------------------------------------------------
// (D1) the 7-vertex model's A_Cq over Z/2^k (as BinaryF2Validation / compute_C_eq_bases assemble them).
template<class Elt>
bool is_identity_fm(const FlatMatrixT<Elt>& M, uint32_t d) {
    if (M.get_rows() != d || M.get_cols() != d) return false;
    for (uint32_t r = 0; r < d; ++r)
        for (uint32_t c = 0; c < d; ++c) if (M.get_entry(r, c) != (r == c ? 1u : 0u)) return false;
    return true;
}
WideMatrix build_ACq(const LocalSystemSimplicialComplexT<uint64_t>& lssc, uint32_t q, uint32_t k) {
    const SimplicialComplex* K = lssc.get_base_K_complex();
    const uint32_t d = lssc.get_representation_dim();
    const uint32_t N_q = K->get_num_q_simplices(q);
    const uint64_t msk = (uint64_t(1) << k) - 1;
    std::vector<const LocalSystemSimplicialComplexT<uint64_t>::SimplexRelation*> live;
    for (const auto& rel : lssc.get_user_defined_simplex_relations()) {
        if (rel.q_dim != q) continue;
        if (rel.s_lhs_idx_in_Kq == rel.s_rhs_idx_in_Kq && is_identity_fm(rel.M, d)) continue;
        live.push_back(&rel);
    }
    WideMatrix A(static_cast<uint32_t>(live.size()) * d, N_q * d, k);
    uint32_t ri = 0;
    for (const auto* rp : live) {
        const auto& rel = *rp;
        for (uint32_t r = 0; r < d; ++r)
            for (uint32_t c = 0; c < d; ++c) {
                const uint64_t mrc = rel.M.get_entry(r, c);
                if (rel.s_lhs_idx_in_Kq == rel.s_rhs_idx_in_Kq) {
                    A.entry(ri * d + r, rel.s_lhs_idx_in_Kq * d + c)[0] = ((r == c ? 1u : 0u) - mrc) & msk;
                } else {
                    if (r == c) A.entry(ri * d + r, rel.s_lhs_idx_in_Kq * d + c)[0] = 1;
                    A.entry(ri * d + r, rel.s_rhs_idx_in_Kq * d + c)[0] = (uint64_t(0) - mrc) & msk;
                }
            }
        ++ri;
    }
    return A;
}

// (D2) exact rho_n(g): entry(k, j) = the coefficient of y^k in (a + c y)^(n-j) (b + d y)^j (the
// ComputeSymmetricPowerMatrixRingOracle formula).
std::vector<BigInt> poly_pow(int64_t a, int64_t c, uint32_t e) {   // (a + c y)^e
    std::vector<BigInt> p(1, BigInt(1));
    for (uint32_t t = 0; t < e; ++t) {
        std::vector<BigInt> q(p.size() + 1, BigInt(0));
        for (size_t i = 0; i < p.size(); ++i) { q[i] += p[i] * BigInt(a); q[i + 1] += p[i] * BigInt(c); }
        p.swap(q);
    }
    return p;
}
ref6::BigMat rho_exact(const hecke::Mat22& g, uint32_t n) {
    const uint32_t d = n + 1;
    ref6::BigMat M(d, d);
    for (uint32_t j = 0; j <= n; ++j) {
        const std::vector<BigInt> p1 = poly_pow(g.a, g.c, n - j), p2 = poly_pow(g.b, g.d, j);
        for (size_t x = 0; x < p1.size(); ++x)
            for (size_t y = 0; y < p2.size(); ++y) M.at(static_cast<uint32_t>(x + y), j) += p1[x] * p2[y];
    }
    return M;
}
ref6::BigMat bmul(const ref6::BigMat& A, const ref6::BigMat& B) {
    ref6::BigMat C(A.rows, B.cols);
    for (uint32_t i = 0; i < A.rows; ++i)
        for (uint32_t t = 0; t < A.cols; ++t) {
            if (A.at(i, t).is_zero()) continue;
            for (uint32_t j = 0; j < B.cols; ++j) C.at(i, j) += A.at(i, t) * B.at(t, j);
        }
    return C;
}
ref6::BigMat bident(uint32_t d) { ref6::BigMat I(d, d); for (uint32_t i = 0; i < d; ++i) I.at(i, i) = 1; return I; }
ref6::BigMat badd(const ref6::BigMat& A, const ref6::BigMat& B, int64_t sb = 1) {
    ref6::BigMat C = A;
    for (size_t x = 0; x < C.a.size(); ++x) C.a[x] += B.a[x] * BigInt(sb);
    return C;
}
void bput(ref6::BigMat& T, const ref6::BigMat& B, uint32_t r0, uint32_t c0, int64_t sgn = 1) {
    for (uint32_t i = 0; i < B.rows; ++i)
        for (uint32_t j = 0; j < B.cols; ++j) T.at(r0 + i, c0 + j) = B.at(i, j) * BigInt(sgn);
}
struct Amalgam { ref6::BigMat d0, z1; };
Amalgam amalgam(uint32_t n) {
    const uint32_t d = n + 1;
    const ref6::BigMat S = rho_exact(hecke::Mat22::S(), n), U = rho_exact(hecke::Mat22::ST(), n), I = bident(d);
    const ref6::BigMat S2 = bmul(S, S), S3 = bmul(S2, S), U2 = bmul(U, U), U3 = bmul(U2, U), U4 = bmul(U3, U), U5 = bmul(U4, U);
    Amalgam am;
    am.d0 = ref6::BigMat(2 * d, d);
    bput(am.d0, badd(S, I, -1), 0, 0);
    bput(am.d0, badd(U, I, -1), d, 0);
    const ref6::BigMat NS = badd(badd(badd(I, S), S2), S3);
    const ref6::BigMat NU = badd(badd(badd(badd(badd(I, U), U2), U3), U4), U5);
    am.z1 = ref6::BigMat(3 * d, 2 * d);
    bput(am.z1, NS, 0, 0);
    bput(am.z1, NU, d, d);
    bput(am.z1, badd(I, S), 2 * d, 0);
    bput(am.z1, badd(badd(I, U), U2), 2 * d, d, -1);
    return am;
}

uint32_t part_amalgam() {
    uint32_t count = 0;
    // (D1)
    const SimplicialComplex K = CreateSL2ZFundamentalDomainTriangulation();
    const uint32_t n_list[] = {1, 2, 3, 4, 5, 8, 11, 16, 24, 33};
    const uint32_t k_list[] = {2, 4, 8, 16, 31, 32, 63};
    for (uint32_t k : k_list) {
        const engine::ModCtx<uint64_t> ctx = engine::ModCtx<uint64_t>::prime_power(2, k);
        for (uint32_t n : n_list) {
            const LocalSystemSimplicialComplexT<uint64_t> lssc(K, hecke::GenerateSL2ZRelationsRing<uint64_t>(K, n, ctx), n + 1, ctx.p);
            for (uint32_t q = 0; q <= 2; ++q) {
                const WideMatrix A = build_ACq(lssc, q, k);
                if (A.rows == 0 || A.cols == 0) continue;
                Opts o;
                o.naive = static_cast<uint64_t>(A.rows) * A.cols <= 20000;
                o.kernel = static_cast<uint64_t>(A.rows) * A.cols <= 20000;
                o.engine_u32 = k <= 31 && n <= 8;
                o.ladic_small_k = n <= 2;
                o.paths = n == 5;
                o.max_r5 = 1;
                compare_matrix("A_C" + str(q) + " n_sym=" + str(n), A, o);
                ++count;
            }
        }
    }
    // (D2) rho_n cross-check against the production composite route
    {
        const engine::ModCtx<uint64_t> c63 = engine::ModCtx<uint64_t>::prime_power(2, 63);
        bool ok = true;
        for (uint32_t n : {1u, 2u, 5u, 12u, 33u, 64u})
            for (const hecke::Mat22& g : {hecke::Mat22::S(), hecke::Mat22::ST(), hecke::Mat22::T()}) {
                const WideMatrix mine = ref6::to_wide(rho_exact(g, n), 63);
                const FlatMatrixT<uint64_t> prod = hecke::RhoOfExactRing<uint64_t>(c63, g, n);
                for (uint32_t i = 0; i <= n; ++i)
                    for (uint32_t j = 0; j <= n; ++j) ok = ok && mine.entry(i, j)[0] == prod.get_entry(i, j);
            }
        check(ok, "exact rho_n(S), rho_n(ST), rho_n(T) mod 2^63 == hecke::RhoOfExactRing (n = 1, 2, 5, 12, 33, 64)");
    }
    const uint32_t kx[] = {8, 40, 63, 64, 65, 100, 128, 129, 256, 600};
    for (uint32_t n : {1u, 2u, 3u, 5u, 8u, 16u, 33u, 64u}) {
        const Amalgam am = amalgam(n);
        for (int which = 0; which < 2; ++which) {
            const ref6::BigMat& M = which ? am.z1 : am.d0;
            const std::string nm = which ? "Z1(cocycle rel) " : "d0(coboundary) ";
            std::vector<uint32_t> prev;
            uint32_t prev_k = 0;
            for (uint32_t k : kx) {
                if (n >= 33 && k > 256) continue;
                const WideMatrix A = ref6::to_wide(M, k);
                Opts o;
                const uint64_t sz = static_cast<uint64_t>(A.rows) * A.cols;
                o.ref6 = n <= 3;
                o.naive = sz <= 20000;
                o.engine = k <= 63 || sz <= 3000;
                o.kernel = sz <= 6000;
                o.max_r5 = 1;
                const r2::SmithResult sm = compare_matrix(nm + "n=" + str(n), A, o);
                // cross-k: Smith over Z/2^k of an integer matrix = its 2-adic Smith valuations capped at k
                if (prev_k) {
                    std::vector<uint32_t> cut;
                    for (uint32_t x : sm.sigma) if (x < prev_k) cut.push_back(x);
                    check(cut == prev, nm + "n=" + str(n) + ": Smith valuations at k=" + str(k) + " restrict to those at k=" + str(prev_k));
                }
                prev = sm.sigma;
                prev_k = k;
                ++count;
            }
            if (n == 3 || n == 8 || n == 33)
                std::cout << "  amalgam " << nm << "n=" << n << " " << M.rows << "x" << M.cols << ": Smith valuations over Z/2^"
                          << prev_k << " " << sig_str(prev) << std::endl;
        }
    }
    return count;
}

// ---------------------------------------------------------------------------
// (E) the adaptive rung
// ---------------------------------------------------------------------------
void part_adaptive() {
    // An exact integer matrix P L D R with D = diag(1, 1, 2, 2^5, 2^70, 2^150, 3, 2^9, 1, 1) (over Z; L, R unimodular).
    const uint32_t n = 10;
    const uint32_t ev[n] = {0, 0, 1, 5, 70, 150, 0, 9, 0, 0};
    uint64_t s = 0xADA9ULL;
    ref6::BigMat L = bident(n), R = bident(n), D(n, n);
    for (uint32_t i = 0; i < n; ++i)
        for (uint32_t j = 0; j < n; ++j) {
            const int64_t v = static_cast<int64_t>(splitmix(s) % 9) - 4;
            if (j < i) L.at(i, j) = v;
            if (j > i) R.at(i, j) = v;
        }
    for (uint32_t i = 0; i < n; ++i) D.at(i, i) = BigInt::pow(BigInt(2), ev[i]) * BigInt(i == 6 ? 3 : 1);
    const ref6::BigMat A = bmul(bmul(L, D), R);
    std::vector<uint32_t> want(ev, ev + n);
    std::sort(want.begin(), want.end());
    auto produce = [&](uint32_t K) { return ref6::to_wide(A, K); };
    {
        const r2::AdaptiveSmith a = r2::smith_adaptive(produce, 20, 4, 1024, n);
        for (const std::string& l : a.log) std::cout << "  " << l << std::endl;
        check(a.resolved && a.smith.sigma == want, "adaptive (rank hint 10): planted 2^70, 2^150 resolved, sigma " + sig_str(a.smith.sigma));
        check(a.verified && a.zeros == 0, "adaptive (rank hint 10): verified, zeros = 0");
        check(a.reruns == 3 && r2::rung_label(a.rung) == "R5(N=3)",
              "adaptive: R2(u32) -> R3 -> R4 -> R5(N=3), 3 reruns (got " + str(a.reruns) + ", " + r2::rung_label(a.rung) + ")");
    }
    {   // without a rank hint the two large invariants read as zero below K = 32: resolved at once
        const r2::AdaptiveSmith a = r2::smith_adaptive(produce, 20, 4, 1024, -1);
        for (const std::string& l : a.log) std::cout << "  " << l << std::endl;
        check(a.resolved && a.reruns == 0 && a.K == 32, "adaptive (no hint): resolved at R2(u32), K = 32 (zero invariants are not near the cap)");
        // BIN-2 audit R1: the two invisible invariants (2^70, 2^150 >= K) are counted and the run is marked unverified.
        check(a.zeros == 2 && !a.verified && a.log.back().find("zeros=2") != std::string::npos &&
                  a.log.back().find("resolved (unverified: no rank hint)") != std::string::npos,
              "adaptive (no hint): zeros = 2 reported, log says 'resolved (unverified: no rank hint)'");
    }
    {   // a near-cap invariant: 2^62 at K = 64 (guard 4) reruns to R4
        ref6::BigMat D2 = D;
        D2.at(4, 4) = BigInt::pow(BigInt(2), 62);
        D2.at(5, 5) = BigInt(1);
        const ref6::BigMat A2 = bmul(bmul(L, D2), R);
        auto produce2 = [&](uint32_t K) { return ref6::to_wide(A2, K); };
        const r2::AdaptiveSmith a = r2::smith_adaptive(produce2, 40, 4, 1024, -1);
        for (const std::string& l : a.log) std::cout << "  " << l << std::endl;
        check(a.resolved && a.reruns == 1 && a.K == 128 && std::count(a.smith.sigma.begin(), a.smith.sigma.end(), 62u) == 1,
              "adaptive (near cap): 2^62 at K = 64 reruns once to R4(u128)");
    }
    {   // an invariant beyond k_cap stays UNRESOLVED and says so
        const r2::AdaptiveSmith a = r2::smith_adaptive(produce, 20, 4, 128, n);
        check(!a.resolved && a.K == 128, "adaptive: 2^150 beyond k_cap = 128 is reported UNRESOLVED");
    }
}

// ---------------------------------------------------------------------------
// (F) negative controls
// ---------------------------------------------------------------------------
struct CarryDropW4 : r2::Traits<r2::WordN<4>> {   // PLANTED BUG: the product drops every carry
    static E mul(const E& a, const E& b) {
        E r = zero();
        for (uint32_t i = 0; i < 4; ++i)
            for (uint32_t j = 0; i + j < 4; ++j) r.w[i + j] += a.w[i] * b.w[j];   // low halves only
        return r;
    }
    static E inv_odd(const E& u) {
        E x = u;
        const E two = from_u64(2);
        for (uint32_t b = 3; b < kBits; b *= 2) x = mul(x, sub(two, mul(u, x)));
        return x;
    }
    static void axpy(E* r, const E* p, const E& f, size_t n) { for (size_t j = 0; j < n; ++j) r[j] = sub(r[j], mul(f, p[j])); }
    static void gemm_sub(E* C, size_t ldc, uint32_t mrows, const E* L, size_t ldl, const E* U, size_t ldu, uint32_t np,
                         uint32_t ncols) {
        for (uint32_t i = 0; i < mrows; ++i)
            for (uint32_t t = 0; t < np; ++t) axpy(C + i * ldc, U + t * ldu, L[i * ldl + t], ncols);
    }
};
struct ShortNewtonU64 : r2::Traits<uint64_t> {   // PLANTED BUG: one Newton step short (48 correct bits)
    static E inv_odd(E u) {
        E x = u;
        for (uint32_t b = 6; b < kBits; b *= 2) x = mul(x, sub(E(2), mul(u, x)));
        return x;
    }
};

void part_negative() {
    uint64_t s = 0xBAD5ULL;
    uint32_t carry_caught = 0, newton_caught = 0, tries = 0;
    for (uint32_t t = 0; t < 12; ++t) {
        ++tries;
        // A buggy body may also trip an internal ENGINE_CHECK (Debug / Check builds): a throw counts as caught.
        {
            const WideMatrix A = random_wide(24, 24, 256, t % 2 ? Fam::Even : Fam::Uniform, s);
            const r2::Mat<r2::WordN<4>> M = r2::from_wide<r2::WordN<4>>(A);
            const WideHowell good = r2::to_wide(r2::howell<r2::Traits<r2::WordN<4>>>(M, 256));
            bool caught = false;
            try {
                const WideHowell bad = r2::to_wide(r2::howell<CarryDropW4>(M, 256));
                const bool differs = !(good == bad) ||
                                     r2::smith<CarryDropW4>(M, 256).sigma != r2::smith<r2::Traits<r2::WordN<4>>>(M, 256).sigma;
                const bool ref_differs = t < 3 ? !(ref6::howell(A) == bad) : differs;
                caught = differs && ref_differs;
            } catch (const std::logic_error&) { caught = true; }
            if (caught) ++carry_caught;
        }
        {
            const WideMatrix A = random_wide(30, 30, 63, t % 2 ? Fam::Even : Fam::Small, s);
            const r2::Mat<uint64_t> M = r2::from_wide<uint64_t>(A);
            const engine::ModCtx<uint64_t> ctx = engine::ModCtx<uint64_t>::prime_power(2, 63);
            const DenseBuf<uint64_t> B = to_engine<uint64_t>(A);
            const WideHowell eng = from_engine(elim::howell<uint64_t>(ctx, cview(B)), A.cols, 63);
            bool caught = false;
            try {
                caught = !(eng == r2::to_wide(r2::howell<ShortNewtonU64>(M, 63)));
            } catch (const std::logic_error&) { caught = true; }
            if (caught) ++newton_caught;
        }
    }
    std::cout << "  NEGATIVE_CONTROL carry_drop(R5 N=4, k=256) caught=" << carry_caught << "/" << tries
              << " short_newton(R3, k=63, vs engine) caught=" << newton_caught << "/" << tries << std::endl;
    check(carry_caught == tries, "negative control: the carry-dropping R5 product is caught on every matrix");
    check(newton_caught == tries, "negative control: the one-step-short Newton inverse is caught by the engine comparison on every matrix");
    {   // the comparison itself sees a single corrupted bit
        const WideMatrix A = random_wide(20, 20, 40, Fam::Uniform, s);
        WideHowell h = r2::howell_wide(A);
        const engine::ModCtx<uint64_t> ctx = engine::ModCtx<uint64_t>::prime_power(2, 40);
        const WideHowell e = from_engine(elim::howell<uint64_t>(ctx, cview(to_engine<uint64_t>(A))), A.cols, 40);
        const bool same_before = h == e;
        h.H.entry(h.H.rows - 1, h.H.cols - 1)[0] ^= uint64_t(1) << 39;
        check(same_before && !(h == e), "negative control: one flipped bit of the Howell form is caught");
    }
}

// ---------------------------------------------------------------------------
// (G) timing (reported, never gated)
// ---------------------------------------------------------------------------
// Returns the Smith time.  t_ref > 0: printed as the ratio to the engine's field RREF (ref = "field")
// or to R3 at the same n (ref = "R3", with the plan's N (N + 1) / 2 word-product model beside it).
template<class E>
double time_rung(const char* label, uint32_t k, uint32_t n, double t_ref, const char* ref, double model = 0) {
    uint64_t s = 0x7173ULL + n + k;
    const WideMatrix A = random_wide(n, n, k, Fam::Uniform, s);
    const r2::Mat<E> M = r2::from_wide<E>(A);
    double ts = 1e30, th = 1e30;
    r2::SmithResult sm;
    for (int rep = 0; rep < 2; ++rep) {
        double t = now_s();
        sm = r2::smith<r2::Traits<E>>(M, k);
        ts = std::min(ts, now_s() - t);
        if (rep == 0) {
            t = now_s();
            (void)r2::howell<r2::Traits<E>>(M, k);
            th = now_s() - t;
        }
    }
    char buf[480];
    if (t_ref <= 0)
        std::snprintf(buf, sizeof(buf), "RING2K_TIMING rung=%s k=%u n=%u path=%s smith_s=%.4f howell_s=%.4f rank=%zu", label, k, n,
                      r2::kernel_path_name(r2::active_kernel_path()), ts, th, sm.sigma.size());
    else if (std::string(ref) == "field")
        std::snprintf(buf, sizeof(buf),
                      "RING2K_TIMING rung=%s k=%u n=%u path=%s smith_s=%.4f howell_s=%.4f engine_field_rref_s=%.4f "
                      "smith_speedup_vs_field=%.2fx rank=%zu",
                      label, k, n, r2::kernel_path_name(r2::active_kernel_path()), ts, th, t_ref, t_ref / ts, sm.sigma.size());
    else
        std::snprintf(buf, sizeof(buf),
                      "RING2K_TIMING rung=%s k=%u n=%u path=%s smith_s=%.4f howell_s=%.4f R3_smith_s=%.4f cost_vs_R3=%.1fx "
                      "(model N(N+1)/2=%.0f) rank=%zu",
                      label, k, n, r2::kernel_path_name(r2::active_kernel_path()), ts, th, t_ref, ts / t_ref, model, sm.sigma.size());
    std::cout << buf << std::endl;
    return ts;
}
double time_field_rref(uint32_t n) {
    uint64_t s = 0x7173ULL + n;
    const uint32_t p = 134217757u;
    const engine::ModCtx<uint32_t> ctx(p);
    DenseBuf<uint32_t> A(n, n);
    for (uint32_t i = 0; i < n; ++i)
        for (uint32_t j = 0; j < n; ++j) A.at(i, j) = static_cast<uint32_t>(splitmix(s) % p);
    double best = 1e30;
    for (int rep = 0; rep < 2; ++rep) {
        DenseBuf<uint32_t> B = A;
        const double t = now_s();
        (void)elim::rref<uint32_t>(ctx, view(B));
        best = std::min(best, now_s() - t);
    }
    return best;
}
void time_vs_engine(uint32_t k, uint32_t n) {
    uint64_t s = 0x5151ULL + k;
    const WideMatrix A = random_wide(n, n, k, Fam::Even, s);
    const engine::ModCtx<uint64_t> ctx = engine::ModCtx<uint64_t>::prime_power(2, k);
    const DenseBuf<uint64_t> B = to_engine<uint64_t>(A);
    double t = now_s();
    const std::vector<uint32_t> ef = elim::invariant_factors<uint64_t>(ctx, cview(B));
    const double t_einv = now_s() - t;
    t = now_s();
    const elim::HowellResult<uint64_t> eh = elim::howell<uint64_t>(ctx, cview(B));
    const double t_eh = now_s() - t;
    const RungChoice rc = r2::select_rung(k);
    t = now_s();
    const r2::SmithResult sm = r2::smith_on(rc, A);
    const double t_s = now_s() - t;
    t = now_s();
    const WideHowell h = r2::howell_on(rc, A);
    const double t_h = now_s() - t;
    check(ef == sigma_to_invariant_factors(sm) && from_engine(eh, A.cols, k) == h, "timing k=" + str(k) + " n=" + str(n) + ": results agree");
    char buf[400];
    std::snprintf(buf, sizeof(buf),
                  "RING2K_ENGINE_TIMING k=%u n=%u rung=%s engine_invariant_factors_s=%.4f ring2k_smith_s=%.4f speedup=%.1fx "
                  "engine_howell_s=%.4f ring2k_howell_s=%.4f speedup=%.1fx",
                  k, n, r2::rung_label(rc).c_str(), t_einv, t_s, t_einv / t_s, t_eh, t_h, t_eh / t_h);
    std::cout << buf << std::endl;
}
void part_timing() {
    const uint32_t n = 1024;
    const double tf = time_field_rref(n);
    std::printf("RING2K_TIMING engine_field_rref p=134217757 n=%u s=%.4f (the blocked mod-p reference)\n", n, tf);
    time_rung<uint8_t>("R1(u8)", 8, n, tf, "field");
    time_rung<uint16_t>("R2(u16)", 16, n, tf, "field");
    time_rung<uint32_t>("R2(u32)", 32, n, tf, "field");
    time_rung<uint64_t>("R3(u64)", 64, n, tf, "field");
    const double r3_512 = time_rung<uint64_t>("R3(u64)", 64, 512, 0, "");
    time_rung<u128>("R4(u128)", 128, 512, r3_512, "R3", 3);
    const double r3_256 = time_rung<uint64_t>("R3(u64)", 64, 256, 0, "");
    time_rung<r2::WordN<4>>("R5(N=4)", 256, 256, r3_256, "R3", 10);
    const double r3_128 = time_rung<uint64_t>("R3(u64)", 64, 128, 0, "");
    time_rung<r2::WordN<16>>("R5(N=16)", 1024, 128, r3_128, "R3", 136);
    time_vs_engine(32, 512);
    time_vs_engine(63, 512);
}

} // namespace

uint32_t Ring2kLadderValidation() {
    const double t0 = now_s();
    g_fail = 0;
    g_checks = 0;
    g_t = Tally{};
    std::cout << "=== Ring2kLadderValidation (engine::ring2k, the Z/2^k ladder R0-R5, vs the engine, engine::ladic and R6) ==="
              << std::endl;
    std::string paths;
    for (r2::KernelPath p : {r2::KernelPath::Portable, r2::KernelPath::Avx2, r2::KernelPath::Avx512})
        if (r2::kernel_path_available(p)) paths += std::string(paths.empty() ? "" : ",") + r2::kernel_path_name(p);
    std::cout << "  kernel paths available: " << paths << " (auto = " << r2::kernel_path_name(r2::active_kernel_path()) << ")"
              << std::endl;
    uint32_t n_random = 0, n_amalgam = 0;
    try {
        part_rungs();
        part_fuzz();
        std::cout << "  rung table + wraparound fuzz: " << g_checks << " checks, " << g_fail << " failing" << std::endl;
        n_random = part_random();
        std::cout << "  random + planted: " << n_random << " matrices (" << g_checks << " checks so far, " << g_fail << " failing)"
                  << std::endl;
        n_amalgam = part_amalgam();
        std::cout << "  amalgam (A_Cq over Z/2^k; exact Sym^n d0 / Z1): " << n_amalgam << " matrices (" << g_checks
                  << " checks so far, " << g_fail << " failing)" << std::endl;
        part_adaptive();
        part_negative();
        part_timing();
    } catch (const std::exception& e) {
        check(false, std::string("exception: ") + e.what());
    }
    r2::set_kernel_path(r2::KernelPath::Auto);
    const double secs = now_s() - t0;
    std::cout << "  comparisons: matrices=" << g_t.matrices << " other-rung runs=" << g_t.rung_runs << " engine(ModCtx)=" << g_t.engine_cmp
              << " engine(ladic)=" << g_t.ladic_cmp << " R6=" << g_t.ref6_cmp << " unblocked=" << g_t.naive_cmp << std::endl;
    std::cout << "RING2K_LADDER rungs=R0,R1,R2,R3,R4,R5,R6 random=" << n_random << " amalgam=" << n_amalgam << " paths=" << paths
              << " checks=" << g_checks << " failing=" << g_fail << " verdict=" << (g_fail == 0 ? "OK" : "FAIL") << std::endl;
    std::cout << "  " << secs << " s" << std::endl;
    std::cout << (g_fail == 0 ? "Ring2kLadderValidation: PASS" : "Ring2kLadderValidation: FAIL") << std::endl;
    return g_fail;
}
