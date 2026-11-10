// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// DESCENT D5: the tree route of the ell-local bulk pieces.  See DescentTree.h; [C] = docs/latex/gl2_contract.tex, cited by
// label (prop:isomodeltree, prop:certifiedeell).
#include "DescentTree.h"
#include "DescentParAsm.h"
#include "DescentAssembly.h"
#include "DescentSmithGpu.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <mutex>
#include <numeric>

#include "../../../ModArithmetic/Binary/Ring2k.h"
#include "../../../ModArithmetic/Binary/Ring3k.h"
#include "../../../ModArithmetic/Binary/SmithPar.h"
#include "../../../ModArithmetic/DenseElim.h"
#include "../../../ModArithmetic/DenseView.h"
#include "../../../ModArithmetic/EngineCheck.h"

namespace integral_descent {

using engine::DenseBuf;
using integral_model::TreeModule;
using U32Ctx = engine::ModCtx<uint32_t>;
using clk = std::chrono::steady_clock;

namespace {
double since(clk::time_point t0) { return std::chrono::duration<double>(clk::now() - t0).count(); }

// (S24 CL-5) HECKE_DESCENT_TREE on EnvLever: "" = unset; an invalid value warns once and reads as the default (source=default).
engine::lever::EnvLever g_tree("HECKE_DESCENT_TREE",
                               engine::lever::Policy{true, engine::lever::Empty::Unset, engine::lever::Match::Exact01, engine::lever::Cache::Once, "WARNING: HECKE_DESCENT_TREE=%s is neither 0 nor 1; the tree route stays on (default)\n", "default", "env", "default"});

// Smith valuations of a row-major matrix over Z/ell^K on the native ring engines (consumes the store).
std::vector<uint32_t> smith_rowmajor(const U32Ctx& ctx, std::vector<uint32_t>&& a, uint32_t rows, uint32_t cols) {
    if (rows == 0 || cols == 0) return {};
    const engine::smithpar::ScopedEnable mt;   // (S24 PAR-2K) the threaded ring2k / ring3k Smith (HECKE_DESCENT_SMITH_MT)
    const uint32_t ell = static_cast<uint32_t>(ctx.ell), K = ctx.k;
    if (ell == 2) {
        engine::ring2k::Mat<uint32_t> W;
        W.rows = rows;
        W.cols = cols;
        W.a = std::move(a);
        return engine::ring2k::smith<engine::ring2k::Traits<uint32_t>>(std::move(W), K).sigma;
    }
    ENGINE_CHECK_MSG(ell == 3 && K >= 2 && K <= engine::ring3k::kMaxK, "smith_rowmajor: ell in {2, 3}");
    return engine::ring3k::smith(DenseBuf<uint32_t>::adopt(std::move(a), rows, cols), K).sigma;
}

uint32_t rank_rowmajor(const U32Ctx& F, std::vector<uint32_t>&& a, uint32_t rows, uint32_t cols) {
    if (rows == 0 || cols == 0) return 0;
    DenseBuf<uint32_t> W = DenseBuf<uint32_t>::adopt(std::move(a), rows, cols);
    return engine::elim::rref(F, engine::view(W)).rank;
}

// Every denominator of the closed forms divides a power of q_ell (C-D6 (1)(d), as CheckDenominators on Iso^(ell)).
uint64_t check_tree_denominators(uint32_t ell) {
    using namespace integral_model::tree_elt;
    const uint32_t q = ell == 2 ? 3u : 4u;
    const std::vector<integral_model::QGroupRingElt> used =
        ell == 2 ? std::vector<integral_model::QGroupRingElt>{eU2(), NS(), one(), S(), c()}
                 : std::vector<integral_model::QGroupRingElt>{eS4(), ec(), one(), U()};
    uint64_t n = 0;
    for (const auto& a : used) {
        for (const auto& kv : a.terms) {
            int64_t den = kv.second.den;
            for (int i = 0; i < 8 && den > 1; ++i) den /= static_cast<int64_t>(std::gcd<uint64_t>(static_cast<uint64_t>(den), q));
            ENGINE_CHECK_MSG(den == 1, "RunTreeLocalPiece: a coefficient's denominator does not divide a power of q_ell");
        }
        ++n;
    }
    return n;
}
}  // namespace

const engine::lever::EnvLever& DescentTreeLever() { return g_tree; }
bool DescentTreeEnabled() { return g_tree.enabled(); }
std::string DescentTreeSource() { return g_tree.source(); }

uint64_t PredictTreeLocalRss(uint32_t ell, uint32_t D, uint32_t d) {
    // the largest reduced differential: (7/3 D)^2 at 2 ((2)(e), r_{U^2} <= D/3 + d), 2 D x 1.5 D at 3, as uint32_t (4 r c bytes),
    // moved into the Smith engine, whose working store adds ~ 0.0-0.3 of it; the rho memo holds <= 12 block-monomial matrices of
    // D x d entries; + 0.2 GiB.  Factor 5 = 4 r c x 1.25: recalibrated (DESCENT5_predictions.md, amendment 1, committed before the cells
    // it admits) from the first-filed 8 on the measured cells D = 2 340 .. 22 044 (measured 3.97-4.27 r c incl. the base: >= 17 %
    // headroom on every one).
    const uint64_t r = (ell == 2) ? (2ull * D + D / 3 + d) : 2ull * D;
    const uint64_t c = (ell == 2) ? r : (3ull * D) / 2 + d;
    // (PAR-GPU audit L2, L3) + the device route's host memory when it would run (ENGINE_DEVICE=gpu; 0 otherwise, so the CPU
    // prediction is unchanged): the 2-adic Smith LU's L store, panel buffers, pinned staging and (verify mode) the input twin.  The
    // rung is u32 (K <= 30) except the level-1 Sym^n W = 61 route (d == D), u64.
    const uint64_t dev = (ell == 2) ? smith_gpu::PredictHostRssBytes(r, c, d == D ? 8u : 4u) : 0;
    // (integration 3, PAR-GP audit F1) the level-1 Sym^n route (d == D) gets 16 rho-memo blocks instead of 12: at n = 8000 the measured
    // tree-piece growth was 6.48 GiB against 6.37 GiB predicted (n = 6000: 3.63 vs 3.67), i.e. the whole D^2 coefficient was ~2 % short;
    // +4 D^2 x 4 B (+15 % of the level-1 D^2 term) restores >= 10 % headroom.  Raises only; level N (d < D) is unchanged.
    const uint64_t memo = (d == D) ? 16ull : 12ull;
    return 5ull * r * c + memo * 4 * uint64_t(D) * d + (200ull << 20) + dev;
}

uint64_t PredictDescentLevel1Rss(uint32_t n) {
    const uint64_t D = uint64_t(n) + 1;
    // Per route (PAR-L1 audit finding 1).  Old route (lever off, or odd n): the shipped C_7 good prime, 2304 D^2.  Sparse route (even n,
    // default): measured whole-run growth 0.06 / 0.15 / 0.47 GiB at n = 500 / 1000 / 2000 (PARL1_report.md), below the 2-adic tree
    // piece's own bound; the sparse cokernel is budgeted at 32 D^2 bytes on top of it (>= 1.4x headroom at n = 2000).
    // (PAR-GP) the dense route (default since PAR-GP) holds e_S^T, e_ST^T, the residual and one rho block: <= 16 D^2 bytes, inside the
    // same 32 D^2 term (DescentGoodSparseEnabled() covers both the sparse and the dense route).
    const uint64_t tree = PredictTreeLocalRss(2, static_cast<uint32_t>(D), static_cast<uint32_t>(D));
    // (PAR-ASM (b)) the concurrent ell >= 5 Bredon-tree primes: their joint prediction (as many at once as AssembleDescent admits at
    // the default ceiling) on top; one prime at a time (lever off, a device run, one prime) stays inside the tree piece's margin.
    uint64_t ell5 = 0;
    if (par::Enabled(par::Lever::Ell) && !par::DeviceRequested()) {
        const uint32_t m = static_cast<uint32_t>(integral_model::DicksonPrimes(n).size());
        const uint32_t w = par::JointWorkers(m, PredictTreeBredonRss(static_cast<uint32_t>(D)), DescentOptions{}.rss_cap_bytes, par::Threads());
        if (w > 1) ell5 = uint64_t(w) * PredictTreeBredonRss(static_cast<uint32_t>(D));
    }
    if (n % 2 == 0 && DescentGoodSparseEnabled()) return 32ull * D * D + tree + ell5;
    return 2304ull * D * D + tree + ell5;
}

LocalPiece RunTreeLocalPiece(uint32_t ell, const TreeModule<uint32_t>& M, const U32Ctx& ctx, uint32_t q_top,
                             const std::vector<uint32_t>& h, uint32_t rank_prime, const TreeModule<uint32_t>* M_rank,
                             int sym_n, bool force_wide, bool explicit_periodic) {
    const auto t0 = clk::now();
    ENGINE_CHECK_MSG(ell == 2 || ell == 3, "RunTreeLocalPiece: the tree route is at ell in {2, 3}");
    ENGINE_CHECK_MSG(ctx.ell == ell && !ctx.is_field(), "RunTreeLocalPiece: the context must be Z/ell^K, K >= 2, at the model's ell");
    ENGINE_CHECK_MSG(q_top >= 1 && q_top <= 64, "RunTreeLocalPiece: q_top");
    ENGINE_CHECK_MSG(h.size() >= q_top + 1u, "RunTreeLocalPiece: rational dims must cover q = 0 .. q_top");
    ENGINE_CHECK_MSG(!force_wide || (ell == 2 && sym_n >= 0), "RunTreeLocalPiece: the W = 61 route is Sym^n at ell = 2 only");
    if (rank_prime) {
        ENGINE_CHECK_MSG(M_rank && M_rank->dim() == M.dim(), "RunTreeLocalPiece: the rank module must be the same module over F_ell'");
        ENGINE_CHECK_MSG(rank_prime >= 5 && IsPrime(rank_prime) && rank_prime % QEll(ell) != 0 && rank_prime != ell,
                         "RunTreeLocalPiece: the rank prime ell' must be >= 5 and prime to q_ell ([C, prop:isomodeltree](1)(d))");
    }
    const uint32_t D = M.dim();
    LocalPiece out;
    out.route = "tree";
    out.ell = ell;
    out.K = ctx.k;
    out.q_top = q_top;
    out.rank_prime = rank_prime;
    out.denominators_checked = check_tree_denominators(ell);

    const auto tb0 = clk::now();
    const integral_model::TreeLocalForm<uint32_t> T = integral_model::MakeTreeLocalForm(ell, ctx, M, q_top);
    out.build_seconds += since(tb0);
    for (uint32_t q = 0; q <= q_top; ++q) {
        // full layout: (+)_x M over the tree cells of degree q (2: 2, 3, 3, ...; 3: 2, 2, 1, 1, ...)
        const uint32_t cells = ell == 2 ? (q == 0 ? 2u : 3u) : (q <= 1 ? 2u : 1u);
        out.full_rank.push_back(cells * D);
        out.reduced_rank.push_back(T.cochain_rank(q));
    }
    out.tors.assign(q_top + 1, AbelianGroupType{});

    // source (ii): the Betti recursion
    std::vector<uint32_t> derived(q_top, 0);
    {
        uint32_t prev = 0;
        for (uint32_t q = 0; q < q_top; ++q) {
            const int64_t r = int64_t(out.reduced_rank[q]) - int64_t(prev) - int64_t(h[q]);
            ENGINE_CHECK_MSG(r >= 0, "RunTreeLocalPiece: negative derived rank_Q d^q (rational dims inconsistent)");
            derived[q] = static_cast<uint32_t>(r);
            prev = derived[q];
        }
    }
    // the periodicity of (2)(d) is consistent with the recursion: rank d^q = rank d^{rep(q)}
    for (uint32_t q = 0; q < q_top; ++q)
        ENGINE_CHECK_MSG(derived[q] == derived[T.representative(q)],
                         "RunTreeLocalPiece: the Betti recursion breaks the periodicity of [C, prop:isomodeltree](2)(d)");
    // source (i): THIS complex over F_ell'
    std::vector<uint32_t> rank_i;
    if (rank_prime) {
        const auto tr = clk::now();
        const U32Ctx F(rank_prime);
        const integral_model::TreeLocalForm<uint32_t> TF = integral_model::MakeTreeLocalForm(ell, F, *M_rank, q_top);
        for (uint32_t q = 0; q < q_top; ++q) {
            const uint32_t rep = explicit_periodic ? q : TF.representative(q);
            if (rep < q) {
                rank_i.push_back(rank_i[rep]);
                continue;
            }
            rank_i.push_back(rank_rowmajor(F, TF.differential(F, *M_rank, q), TF.cochain_rank(q + 1), TF.cochain_rank(q)));
        }
        // (4)(b) at q = 0: rank P_0 - rank delta^0 over F_ell' = b_0 (P_0's image has rank c_0 over F_ell' as well)
        ENGINE_CHECK_MSG(TF.cochain_rank(0) == out.reduced_rank[0], "RunTreeLocalPiece: c_0 over F_ell' != c_0 over Z/ell^K");
        out.rank_seconds += since(tr);
    }

    for (uint32_t q = 0; q < q_top; ++q) {
        const uint32_t rep = explicit_periodic ? q : T.representative(q);
        if (rep < q) {
            LocalDifferential ld = out.d[rep];
            ld.q = q;
            ld.copy_of = static_cast<int32_t>(rep);
            for (uint32_t a : ld.nonunit) out.tors[q + 1].add_torsion(ell, a);
            out.d.push_back(std::move(ld));
            continue;
        }
        LocalDifferential ld;
        ld.q = q;
        ld.rank_derived = derived[q];
        ld.rank_Q = rank_prime ? rank_i[q] : derived[q];
        if (q == 0) out.dimH0_rank_prime = out.reduced_rank[0] - std::min(out.reduced_rank[0], ld.rank_Q);
        if (ld.rank_Q != ld.rank_derived)
            std::fprintf(stderr, "RunTreeLocalPiece: rank mismatch ell=%u D=%u q=%u rank_F=%u derived=%u\n", ell, D, q, ld.rank_Q,
                         ld.rank_derived);
        ENGINE_CHECK_MSG(ld.rank_Q == ld.rank_derived,
                         "RunTreeLocalPiece: rank_{F_ell'} d^q != c_q - b_q - rank d^{q-1} (ell' does not qualify, or wrong b_q)");
        ld.rows = T.cochain_rank(q + 1);
        ld.cols = T.cochain_rank(q);
        out.peak_matrix_bytes = std::max<uint64_t>(out.peak_matrix_bytes, 2ull * 4 * uint64_t(ld.rows) * ld.cols);
        std::vector<uint32_t> sigma;
        std::vector<uint32_t> A;
        if (!force_wide) {
            const auto tb = clk::now();
            A = T.differential(ctx, M, q);
            out.build_seconds += since(tb);
        }
        std::vector<uint32_t> A_keep;   // the K-route matrix, kept only when the W = 61 route may need to compare with it
        if (!force_wide) {
            if (ell == 2 && sym_n >= 0) A_keep = A;   // Sym^n only: small
            const auto ts = clk::now();
            sigma = smith_rowmajor(ctx, std::move(A), ld.rows, ld.cols);
            out.smith_seconds += since(ts);
            ld.W = ctx.k;
            ENGINE_CHECK_MSG(sigma.size() <= ld.rank_Q, "RunTreeLocalPiece: r_K(d^q) > rank_Q(d^q)");
        }
        if ((force_wide || sigma.size() < ld.rank_Q) && ell == 2 && sym_n >= 0) {
            // [C, prop:certifiedeell](6)'s stopping rule at 2, on this complex ((1)(d)): the O_2 -> Z/2^61 reduction.
            const auto tw = clk::now();
            const auto c64 = engine::ModCtx<uint64_t>::prime_power(2, 61);
            const auto M64 = integral_model::TreeSymModule(c64, static_cast<uint32_t>(sym_n));
            const auto T64 = integral_model::MakeTreeLocalForm(2u, c64, M64, q_top);
            ENGINE_CHECK_MSG(T64.cochain_rank(q) == ld.cols && T64.cochain_rank(q + 1) == ld.rows,
                             "RunTreeLocalPiece: the W = 61 complex has other cochain ranks");
            std::vector<uint64_t> A64 = T64.differential(c64, M64, q);
            if (!force_wide) {
                const uint64_t mask = (uint64_t(1) << ctx.k) - 1;
                for (size_t i = 0; i < A64.size(); ++i)
                    ENGINE_CHECK_MSG((A64[i] & mask) == A_keep[i], "RunTreeLocalPiece: the W = 61 matrix mod 2^K differs");
            }
            engine::ring2k::Mat<uint64_t> W;
            W.rows = ld.rows;
            W.cols = ld.cols;
            W.a = std::move(A64);
            out.build_seconds += since(tw);
            const auto ts = clk::now();
            {
                const engine::smithpar::ScopedEnable mt;   // (S24 PAR-2K)
                sigma = engine::ring2k::smith<engine::ring2k::Traits<uint64_t>>(std::move(W), 61).sigma;
            }
            out.smith_seconds += since(ts);
            ld.W = 61;
            ld.escalated = !force_wide;
            ENGINE_CHECK_MSG(sigma.size() <= ld.rank_Q, "RunTreeLocalPiece: r_K(d^q) > rank_Q(d^q) at W = 61");
        }
        ld.r_K = static_cast<uint32_t>(sigma.size());
        ld.certified = ld.r_K == ld.rank_Q;
        if (ld.certified) {
            for (uint32_t s : sigma)
                if (s >= 1) ld.nonunit.push_back(s);
            std::sort(ld.nonunit.begin(), ld.nonunit.end(), std::greater<uint32_t>());
            if (q >= 1)
                for (uint32_t a : ld.nonunit)
                    ENGINE_CHECK_MSG(a <= (ell == 2 ? 3u : 1u),
                                     "RunTreeLocalPiece: an exponent above v_ell(24) at q >= 1 ([C, prop:certifiedeell](6))");
            for (uint32_t a : ld.nonunit) out.tors[q + 1].add_torsion(ell, a);
        }
        out.d.push_back(std::move(ld));
    }
    out.seconds = since(t0);
    return out;
}

LocalPiece RunTreeLocalPieceSym(uint32_t ell, uint32_t n, uint32_t q_top, const std::vector<uint32_t>& h, uint32_t rank_prime,
                                uint32_t K, bool force_wide, bool explicit_periodic) {
    if (K == 0) K = LocalWidth(ell);
    const U32Ctx ctx = U32Ctx::prime_power(ell, K);
    const TreeModule<uint32_t> M = integral_model::TreeSymModule(ctx, n);
    if (!rank_prime)
        return RunTreeLocalPiece(ell, M, ctx, q_top, h, 0, nullptr, static_cast<int>(n), force_wide, explicit_periodic);
    const TreeModule<uint32_t> MR = integral_model::TreeSymModule(U32Ctx(rank_prime), n);
    return RunTreeLocalPiece(ell, M, ctx, q_top, h, rank_prime, &MR, static_cast<int>(n), force_wide, explicit_periodic);
}

LocalPiece RunTreeLocalPieceN(uint32_t ell, uint32_t N, uint32_t n, uint32_t q_top, const std::vector<uint32_t>& h,
                              uint32_t rank_prime, uint32_t K, bool explicit_periodic) {
    if (K == 0) K = LocalWidth(ell);
    const U32Ctx ctx = U32Ctx::prime_power(ell, K);
    const TreeModule<uint32_t> M = integral_model::TreeInducedSymModule(N, n, ctx);
    if (!rank_prime) return RunTreeLocalPiece(ell, M, ctx, q_top, h, 0, nullptr, -1, false, explicit_periodic);
    const TreeModule<uint32_t> MR = integral_model::TreeInducedSymModule(N, n, U32Ctx(rank_prime));
    return RunTreeLocalPiece(ell, M, ctx, q_top, h, rank_prime, &MR, -1, false, explicit_periodic);
}

BredonSmithRun RunTreeBredonSym(uint32_t n, uint32_t ell, uint32_t K, uint32_t b0) {
    const auto t0 = clk::now();
    ENGINE_CHECK_MSG(ell >= 5 && IsPrime(ell), "RunTreeBredonSym: ell >= 5 prime");
    BredonSmithRun r;
    r.ell = ell;
    r.K = K ? K : integral_model::DefaultKBudget(ell);
    const U32Ctx ctx = U32Ctx::prime_power(ell, r.K);
    const TreeModule<uint32_t> M = integral_model::TreeSymModule(ctx, n);
    const auto tb = clk::now();
    const integral_model::TreeLocalForm<uint32_t> T = integral_model::MakeTreeLocalForm(ell, ctx, M, 1);
    std::vector<uint32_t> A = T.differential(ctx, M, 0);
    r.build_seconds = since(tb);
    r.c0 = T.cochain_rank(0);
    r.c1 = T.cochain_rank(1);
    r.c2 = 0;
    r.rank_Q = r.c0 - b0;
    r.peak_matrix_bytes = 4ull * r.c0 * r.c1;
    const auto ts = clk::now();
    const std::vector<uint32_t> sigma = r.c0 && r.c1 ? SmithLevelsEll(ell, r.K, DenseBuf<uint32_t>::adopt(std::move(A), r.c1, r.c0))
                                                     : std::vector<uint32_t>{};
    r.smith_seconds = since(ts);
    r.r_K = static_cast<uint32_t>(sigma.size());
    r.certified = r.r_K == r.rank_Q;
    if (r.certified)
        for (uint32_t s : sigma)
            if (s >= 1) r.tors1.add_torsion(ell, s);
    r.free_rank1 = r.c1 - r.rank_Q;
    r.seconds = since(t0);
    return r;
}

BadPrimeRun RunTreeBredonLevel1(uint32_t n, uint32_t ell, uint32_t count) {
    const auto t0 = clk::now();
    BadPrimeRun out;
    out.ell = ell;
    out.count = count;
    const BredonSmithRun b = RunTreeBredonSym(n, ell, 0, n == 0 ? 1u : 0u);
    out.K_tried.push_back(b.K);
    out.run_seconds.push_back(b.seconds);
    out.certified = b.certified;
    if (out.certified) out.K = b.K;
    out.tors1 = b.tors1;
    out.free_rank1 = b.free_rank1;
    out.torsion_only_in_H1 = true;   // C^q_{T,ell} = 0 for q >= 2 ((2)(c)): no ell-torsion above H^1, on the model itself
    out.seconds = since(t0);
    return out;
}

}  // namespace integral_descent
