// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// CudaImmaElimValidation.cpp -- see CudaImmaElimValidation.h (slot-22 G2).
#include "CudaImmaElimValidation.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <iostream>
#include <sstream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "CudaCtx.h"
#include "CudaKernels.h"
#include "DenseBuf.h"
#include "DenseView.h"
#include "Kernels.h"
#include "ModCtx.h"
#include "../ModArithmetic/BlockElim.h"
#include "../ModArithmetic/BlockKernels.h"
#include "../ModArithmetic/DenseElim.h"
#include "../Engine/EngineRuntime.h"
#include "../Engine/EngineTestUtil.h"
#include "../Engine/Executor.h"
#include "../Engine/TapeElim.h"
#include "LegacyConfig.h"          // (S24 CL-R2) RunLegacyArm
#include "RetiredDeviceLevers.h"   // (S24 CL-R2) the EFFECTIVE line

namespace {

using namespace engine;
using engine::tape::test::Rng;
namespace cu = engine::cuda;
namespace tp = engine::tape;

uint64_t g_checks = 0, g_fail = 0;
void check(bool ok, const std::string& what) {
    ++g_checks;
    if (!ok) {
        ++g_fail;
        if (g_fail <= 40) std::cout << "  [FAIL] " << what << std::endl;
    }
}
template<class T> std::string str(T v) { return std::to_string(v); }

// A scoped environment variable (restored on exit). The validation is single-threaded at these points.
struct ScopedEnv {
    std::string name; bool had = false; std::string old;
    ScopedEnv(const char* n, const char* v) : name(n) {
        const char* e = std::getenv(n);
        if (e != nullptr) { had = true; old = e; }
        if (v != nullptr) setenv(n, v, 1); else unsetenv(n);
    }
    ~ScopedEnv() { if (had) setenv(name.c_str(), old.c_str(), 1); else unsetenv(name.c_str()); }
};

using Buf = DenseBuf<uint32_t>;

// ---- inputs ------------------------------------------------------------------------
enum class Kind { Random, Planted, Boundaries, Sparse, Extreme, Zero, Identity, DupRows };
const char* kname(Kind k) {
    switch (k) {
        case Kind::Random: return "random";
        case Kind::Planted: return "planted";
        case Kind::Boundaries: return "boundaries";
        case Kind::Sparse: return "sparse";
        case Kind::Extreme: return "extreme";
        case Kind::Zero: return "zero";
        case Kind::Identity: return "identity";
        default: return "dup-rows";
    }
}
Buf make_matrix(Rng& rng, uint32_t m, uint32_t n, uint32_t p, Kind kind, uint32_t rank_req = 0) {
    const ModCtx<uint32_t> ctx(p);
    Buf M(m, n);
    switch (kind) {
        case Kind::Random: case Kind::Boundaries: case Kind::DupRows:
            for (uint32_t i = 0; i < m; ++i) for (uint32_t j = 0; j < n; ++j) M.at(i, j) = rng.elt<uint32_t>(p);
            break;
        case Kind::Planted: {
            const uint32_t r = std::max<uint32_t>(1, std::min({rank_req, m, n}));
            Buf L = tp::test::random_buf<uint32_t>(rng, m, r, p), R = tp::test::random_buf<uint32_t>(rng, r, n, p);
            kernels::gemm<uint32_t>(ctx, cview(L), cview(R), view(M), kernels::GemmOp::Set);
            break;
        }
        case Kind::Sparse:
            for (uint32_t i = 0; i < m; ++i)
                for (uint32_t j = 0; j < n; ++j) {
                    const uint64_t z = rng.next();
                    if (z % 1000 < 3) M.at(i, j) = ((z >> 20) % 3 == 0) ? p - 1 : ((z >> 20) % 3 == 1 ? 1u : static_cast<uint32_t>((z >> 32) % p));
                }
            break;
        case Kind::Extreme:   // every entry +-floor(p/2) (the int8 extremes), sign at random
            for (uint32_t i = 0; i < m; ++i)
                for (uint32_t j = 0; j < n; ++j) M.at(i, j) = (rng.next() & 1) ? p / 2 : p - p / 2;
            break;
        case Kind::Zero: break;
        case Kind::Identity:
            for (uint32_t i = 0; i < std::min(m, n); ++i) M.at(i, i) = 1;
            break;
    }
    if (kind == Kind::Boundaries) {   // zero columns at base / panel boundaries, duplicated columns, a zero row
        for (uint32_t c : {0u, 15u, 16u, 31u, 32u, 63u, 64u, 255u, 256u, 257u, 511u, 512u})
            if (c < n) for (uint32_t i = 0; i < m; ++i) M.at(i, c) = 0;
        for (uint32_t t = 0; t < std::max<uint32_t>(2, n / 40); ++t) {
            uint32_t c = rng.below(n), d = rng.below(n);
            if (c > d) std::swap(c, d);
            if (c == d) continue;
            for (uint32_t i = 0; i < m; ++i) M.at(i, d) = M.at(i, c);
        }
        if (m > 3) for (uint32_t j = 0; j < n; ++j) M.at(m / 2, j) = 0;
    }
    if (kind == Kind::DupRows) {      // duplicated and scaled rows: the pivot-row choice matters
        for (uint32_t t = 0; t < std::max<uint32_t>(2, m / 3); ++t) {
            const uint32_t a = rng.below(m), b = rng.below(m);
            if (a == b) continue;
            const uint32_t s = rng.elt<uint32_t>(p);
            for (uint32_t j = 0; j < n; ++j) M.at(b, j) = ctx.mul(s, M.at(a, j));
        }
    }
    return M;
}

// The host engine under the thread-local Cpu override (the hook can never fire here).
elim::RrefResult cpu_rref(const ModCtx<uint32_t>& ctx, Buf& A) {
    tp::ScopedThreadDevice cpu(tp::ThreadDevice::Cpu);
    return elim::rref<uint32_t>(ctx, view(A));
}
// The row rank profile of A: the column rank profile of A^T.
std::vector<uint32_t> row_rank_profile(const ModCtx<uint32_t>& ctx, const Buf& A) {
    Buf T(A.cols, A.rows);
    for (uint32_t i = 0; i < A.rows; ++i) for (uint32_t j = 0; j < A.cols; ++j) T.at(j, i) = A.at(i, j);
    tp::ScopedThreadDevice cpu(tp::ThreadDevice::Cpu);
    return elim::column_rank_profile<uint32_t>(ctx, view(T)).pivots;
}
// Scalar reference of the pivot rule on the unpermuted rows: per column, the unused row of
// smallest index with a non-zero Schur-complement entry. largest == true: the wrong rule (NC).
std::vector<uint32_t> rule_reference(const ModCtx<uint32_t>& ctx, const Buf& A, bool largest = false) {
    const uint32_t m = A.rows, n = A.cols;
    std::vector<uint32_t> M(static_cast<size_t>(m) * n);
    for (uint32_t i = 0; i < m; ++i) for (uint32_t j = 0; j < n; ++j) M[static_cast<size_t>(i) * n + j] = A.at(i, j);
    std::vector<char> used(m, 0);
    std::vector<uint32_t> rows;
    for (uint32_t c = 0; c < n; ++c) {
        int64_t pr = -1;
        for (uint32_t i = 0; i < m; ++i) {
            if (used[i] || M[static_cast<size_t>(i) * n + c] == 0) continue;
            pr = i;
            if (!largest) break;
        }
        if (pr < 0) continue;
        used[pr] = 1;
        rows.push_back(static_cast<uint32_t>(pr));
        const uint32_t* prow = &M[static_cast<size_t>(pr) * n];
        const uint32_t inv = ctx.inv(prow[c]);
        for (uint32_t i = 0; i < m; ++i) {
            if (used[i]) continue;
            uint32_t* row = &M[static_cast<size_t>(i) * n];
            if (row[c] == 0) continue;
            const uint32_t f = ctx.mul(row[c], inv);
            for (uint32_t j = c; j < n; ++j) row[j] = ctx.sub(row[j], ctx.mul(f, prow[j]));
        }
    }
    return rows;
}

// ---- (A) imma_rref directly ----------------------------------------------------------
struct DirectOut { cu::ImmaRrefResult r; Buf R; };
DirectOut device_rref(const ModCtx<uint32_t>& ctx, const Buf& A, uint32_t slab_rows = 0, uint32_t bw = 32, bool want = true) {
    DirectOut o;
    const uint32_t m = A.rows, n = A.cols;
    cu::ImmaRowSink sink;
    if (want) sink = [&](uint32_t rank) { o.R = Buf(rank, n); return view(o.R); };
    o.r = cu::imma_rref(ctx, m, n, [&](uint32_t r0, uint32_t nr) { return cview(A).sub(r0, nr, 0, n); }, sink, slab_rows, bw);
    return o;
}

void arm_a_case(const ModCtx<uint32_t>& ctx, Rng& rng, uint32_t m, uint32_t n, Kind kind, uint32_t rank_req, uint32_t bw,
                uint32_t slab_rows, bool windowed) {
    const uint32_t p = static_cast<uint32_t>(ctx.p);
    const std::string tag = "(A) p=" + str(p) + " " + str(m) + "x" + str(n) + " " + kname(kind) +
                            (rank_req ? " r=" + str(rank_req) : "") + " W=" + str(bw) + (slab_rows ? " slab=" + str(slab_rows) : "") +
                            (windowed ? " windowed" : "");
    const Buf A = make_matrix(rng, m, n, p, kind, rank_req);
    Buf ref(A);
    const elim::RrefResult er = cpu_rref(ctx, ref);
    DirectOut d;
    if (!windowed) {
        d = device_rref(ctx, A, slab_rows, bw);
    } else {   // source and sink are windows of wider buffers (ld > n): the row-wise / staged paths
        Buf W(m, n + 37), Rw;
        for (uint32_t i = 0; i < m; ++i) for (uint32_t j = 0; j < n + 37; ++j) W.at(i, j) = (j >= 5 && j < 5 + n) ? A.at(i, j - 5) : (p - 1);
        d.r = cu::imma_rref(ctx, m, n,
                            [&](uint32_t r0, uint32_t nr) { return cview(W).sub(r0, nr, 5, n); },
                            [&](uint32_t rank) { Rw = Buf(rank, n + 11); return view(Rw).sub(0, rank, 3, n); },
                            slab_rows, bw);
        d.R = Buf(d.r.rank, n);
        for (uint32_t i = 0; i < d.r.rank; ++i) for (uint32_t j = 0; j < n; ++j) d.R.at(i, j) = Rw.at(i, j + 3);
        bool border = true;   // the sink's columns outside the window are untouched
        for (uint32_t i = 0; i < d.r.rank && border; ++i) for (uint32_t j : {0u, 1u, 2u, n + 3, n + 10}) border = border && Rw.at(i, j) == 0;
        check(border, tag + ": sink writes stay inside the window");
    }
    check(d.r.rank == er.rank, tag + ": rank " + str(d.r.rank) + " vs " + str(er.rank));
    check(d.r.pivots == er.pivots, tag + ": pivots");
    bool same = d.R.rows == er.rank;
    for (uint32_t i = 0; same && i < er.rank; ++i) same = std::memcmp(&d.R.at(i, 0), &ref.at(i, 0), sizeof(uint32_t) * n) == 0;
    check(same, tag + ": RREF bytes");
    bool lower_zero = true;   // the host engine's rows below the rank are zero (the D5b convention both sides share)
    for (uint32_t i = er.rank; lower_zero && i < m; ++i) for (uint32_t j = 0; j < n; ++j) lower_zero = lower_zero && ref.at(i, j) == 0;
    check(lower_zero, tag + ": host RREF zero rows");
    std::vector<uint32_t> sorted = d.r.pivot_rows;
    std::sort(sorted.begin(), sorted.end());
    check(sorted == row_rank_profile(ctx, A), tag + ": pivot_rows set == row rank profile");
    if (static_cast<uint64_t>(m) * n <= 400000u) check(d.r.pivot_rows == rule_reference(ctx, A), tag + ": pivot_rows == the smallest-row rule");
}

void arm_a() {
    Rng rng(0xA11CE);
    for (uint32_t p : {2u, 3u, 5u, 13u, 127u}) {
        const ModCtx<uint32_t> ctx(p);
        const uint32_t bw = (p == 3u || p == 127u) ? 16u : 32u;
        arm_a_case(ctx, rng, 1, 1, Kind::Random, 0, bw, 0, false);
        arm_a_case(ctx, rng, 1, 77, Kind::Random, 0, bw, 0, false);
        arm_a_case(ctx, rng, 77, 1, Kind::Random, 0, bw, 0, false);
        arm_a_case(ctx, rng, 33, 33, Kind::Zero, 0, bw, 0, false);
        arm_a_case(ctx, rng, 40, 33, Kind::Identity, 0, bw, 0, false);
        arm_a_case(ctx, rng, 300, 400, Kind::Random, 0, bw, 0, false);
        arm_a_case(ctx, rng, 400, 300, Kind::Boundaries, 0, 32, 0, false);
        arm_a_case(ctx, rng, 290, 520, Kind::Boundaries, 0, 16, 7, false);
        arm_a_case(ctx, rng, 260, 300, Kind::DupRows, 0, bw, 0, false);
        arm_a_case(ctx, rng, 300, 280, Kind::Planted, 41, bw, 0, false);
        arm_a_case(ctx, rng, 270, 330, Kind::Extreme, 0, bw, 0, false);
        arm_a_case(ctx, rng, 600, 700, Kind::Sparse, 0, bw, 0, false);
        arm_a_case(ctx, rng, 777, 1301, Kind::Planted, 500, 32, 0, false);
        arm_a_case(ctx, rng, 1000, 700, Kind::Boundaries, 0, 16, 64, true);
    }
    // larger shapes at the census primes' range
    for (uint32_t p : {13u, 127u}) {
        const ModCtx<uint32_t> ctx(p);
        arm_a_case(ctx, rng, 2048, 2048, Kind::Planted, 1900, 32, 0, false);
        arm_a_case(ctx, rng, 2500, 2200, Kind::Boundaries, 0, 32, 0, false);
        arm_a_case(ctx, rng, 4100, 300, Kind::Sparse, 0, 32, 0, false);
        arm_a_case(ctx, rng, 300, 4100, Kind::Random, 0, 16, 0, false);
    }
    // rank / pivots only (empty sink): no RREF download, the same rank and pivots
    {
        const ModCtx<uint32_t> ctx(13);
        const Buf A = make_matrix(rng, 500, 640, 13, Kind::Boundaries);
        Buf ref(A);
        const elim::RrefResult er = cpu_rref(ctx, ref);
        const DirectOut d = device_rref(ctx, A, 0, 32, false);
        check(d.r.rank == er.rank && d.r.pivots == er.pivots, "(A) rank-only sink: rank / pivots");
    }
}

// ---- (B) the tape driver under the Gpu override ----------------------------------------
void arm_b_case(tp::Executor<uint32_t>& ex, Rng& rng, uint32_t p, tp::test::Shape shape, const std::vector<uint32_t>& rd,
                const std::vector<uint32_t>& cd, uint32_t density, const std::string& gname, uint64_t* order_differs) {
    const ModCtx<uint32_t> ctx(p);
    auto pool = std::make_shared<BlockPool<uint32_t>>();
    const BlockMat<uint32_t> A = tp::test::make_input<uint32_t>(rng, pool, p, shape, rd, cd, density);
    const std::string tag = "(B) p=" + str(p) + " " + gname + " " + tp::test::shape_name(shape);
    // every engine::blk reference under the Cpu override (ENGINE_DEVICE may be gpu for the whole process)
    std::unique_ptr<tp::ScopedThreadDevice> scope(new tp::ScopedThreadDevice(tp::ThreadDevice::Cpu));   // reset() before each new scope: LIFO restore
    const blk::BlockRrefResult<uint32_t> ref = blk::rref<uint32_t>(A);
    const BlockMat<uint32_t> k_ref = blk::kernel_basis<uint32_t>(A), i_ref = blk::image_basis<uint32_t>(A);
    scope.reset(); scope.reset(new tp::ScopedThreadDevice(tp::ThreadDevice::Gpu));
    const uint64_t before = tp::device_elim_totals().tape_rrefs;
    tp::TapeElimStats st;
    const blk::BlockRrefResult<uint32_t> got = tp::rref<uint32_t>(ex, A, &st);
    check(tp::device_elim_totals().tape_rrefs == before + 1 && st.device_elims == 1, tag + ": the device route was taken");
    check(got.rank == ref.rank, tag + ": rank");
    check(got.pivots == ref.pivots, tag + ": pivots");
    check(got.pivot_rows == ref.pivot_rows, tag + ": pivot_rows (the driver's per-panel order)");
    check(same_entries(cview(got.R), cview(ref.R)) && got.R.rows == ref.R.rows, tag + ": R");
    if (order_differs != nullptr && got.rank > 0) {   // NC: the raw device order (pivot order) vs the panel order
        Buf D = A.to_dense();
        const DirectOut d = device_rref(ctx, D, 0, 32, false);
        if (d.r.pivot_rows != got.pivot_rows) ++*order_differs;
    }
    const BlockMat<uint32_t> kt = tp::kernel_basis<uint32_t>(ex, A);
    check(blk::equal<uint32_t>(kt, k_ref), tag + ": kernel_basis");
    const BlockMat<uint32_t> it = tp::image_basis<uint32_t>(ex, A);
    check(blk::equal<uint32_t>(it, i_ref), tag + ": image_basis");
    const std::vector<uint32_t> row_dims = A.col_dims();
    const uint32_t a = A.cols();
    if (a == 0) return;
    const uint32_t nr = 1 + rng.below(3), ng = 3 + rng.below(6);
    Buf Rl = tp::test::random_buf<uint32_t>(rng, a, nr, p, Layout::ColMajor);
    Buf G = tp::test::random_buf<uint32_t>(rng, a, ng, p, Layout::ColMajor);
    for (uint32_t i = 0; i < a; ++i) {
        G.at(i, 1) = Rl.at(i, 0);
        if (ng > 3) G.at(i, 3) = ctx.add(G.at(i, 0), ctx.mul(2u % p, G.at(i, 2)));
    }
    scope.reset(); scope.reset(new tp::ScopedThreadDevice(tp::ThreadDevice::Cpu));
    const blk::FactorBasisResult<uint32_t> fr = blk::factor_basis<uint32_t>(A.pool(), p, row_dims, cview(Rl), cview(G), blk::ElimParams{256});
    scope.reset(); scope.reset(new tp::ScopedThreadDevice(tp::ThreadDevice::Gpu));
    const blk::FactorBasisResult<uint32_t> ft = tp::factor_basis<uint32_t>(ex, A.pool(), p, row_dims, cview(Rl), cview(G), 256);
    check(fr.pivots == ft.pivots && fr.independent_relations == ft.independent_relations, tag + ": factor_basis pivots");
    check(blk::equal<uint32_t>(fr.basis, ft.basis) && fr.basis.cols() == ft.basis.cols(), tag + ": factor_basis basis");
    check(same_entries(cview(fr.projector), cview(ft.projector)), tag + ": factor_basis projector");
    const uint32_t ny = 2 + rng.below(3);
    Buf coef = tp::test::random_buf<uint32_t>(rng, ng, ny, p);
    Buf Y(a, ny, Layout::ColMajor);
    kernels::gemm<uint32_t>(ctx, cview(G), cview(coef), view(Y), kernels::GemmOp::Set);
    scope.reset(); scope.reset(new tp::ScopedThreadDevice(tp::ThreadDevice::Cpu));
    const Buf sr = blk::solve_multi_rhs<uint32_t>(A.pool(), p, row_dims, cview(G), cview(Y), blk::ElimParams{256});
    scope.reset(); scope.reset(new tp::ScopedThreadDevice(tp::ThreadDevice::Gpu));
    const Buf sg = tp::solve_multi_rhs<uint32_t>(ex, A.pool(), p, row_dims, cview(G), cview(Y), 256);
    check(same_entries(cview(sr), cview(sg)), tag + ": solve_multi_rhs");
    if (fr.pivots.size() < a) {
        Buf Yo(Y);
        for (uint32_t i = 0; i < a; ++i) Yo.at(i, 0) = rng.elt<uint32_t>(p);
        std::string m1, m2;
        scope.reset(); scope.reset(new tp::ScopedThreadDevice(tp::ThreadDevice::Cpu));
        try { (void)blk::solve_multi_rhs<uint32_t>(A.pool(), p, row_dims, cview(G), cview(Yo), blk::ElimParams{256}); } catch (const std::runtime_error& e) { m1 = e.what(); }
        scope.reset(); scope.reset(new tp::ScopedThreadDevice(tp::ThreadDevice::Gpu));
        try { (void)tp::solve_multi_rhs<uint32_t>(ex, A.pool(), p, row_dims, cview(G), cview(Yo), 256); } catch (const std::runtime_error& e) { m2 = e.what(); }
        check(m1 == m2, tag + ": out-of-span verdict");
    }
}

void arm_b() {
    ScopedEnv min0("ENGINE_GPU_ELIM_MIN", "0");
    tp::Executor<uint32_t> ex;
    Rng rng(0xB0B);
    uint64_t order_differs = 0, order_cases = 0;
    struct Grid { std::vector<uint32_t> rd, cd; uint32_t density; const char* name; };
    const std::vector<Grid> grids{
        {{4, 4, 4}, {4, 4, 4, 4}, 60, "3x4 d=4"},
        {{3, 7, 2}, {5, 1, 6}, 70, "non-uniform"},
        {{3, 0, 2}, {5, 0, 6}, 70, "zero-width"},
        {{1, 1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, 80, "scalar blocks"},
        {std::vector<uint32_t>(6, 17), std::vector<uint32_t>(7, 17), 25, "6x7 d=17"},
        {std::vector<uint32_t>(13, 17), std::vector<uint32_t>(7, 17), 22, "13x7 d=17"},
        {std::vector<uint32_t>(6, 65), std::vector<uint32_t>(7, 65), 25, "6x7 d=65"},
        {{129, 129}, std::vector<uint32_t>(11, 129), 25, "2x11 d=129"},
        {std::vector<uint32_t>(13, 129), std::vector<uint32_t>(7, 129), 18, "13x7 d=129"},
    };
    for (uint32_t p : {2u, 13u, 127u})
        for (const Grid& g : grids)
            for (tp::test::Shape sh : {tp::test::Shape::CqLike, tp::test::Shape::ZqLike, tp::test::Shape::Generic, tp::test::Shape::Dependent}) {
                const bool nc = (g.rd.size() >= 6 && g.rd[0] >= 17);
                if (nc) ++order_cases;
                arm_b_case(ex, rng, p, sh, g.rd, g.cd, g.density, g.name, nc ? &order_differs : nullptr);
            }
    // one-column panels: the driver's order is then the pivot order itself
    for (uint32_t p : {3u, 13u}) {
        auto pool = std::make_shared<BlockPool<uint32_t>>();
        const ModCtx<uint32_t> ctx(p);
        BlockBuilder<uint32_t> b(pool, std::vector<uint32_t>{90}, std::vector<uint32_t>(70, 1), p);
        Buf M = make_matrix(rng, 90, 70, p, Kind::DupRows);
        for (uint32_t c = 0; c < 70; ++c) {
            Buf col(90, 1);
            for (uint32_t i = 0; i < 90; ++i) col.at(i, 0) = M.at(i, c);
            b.set(0, c, BlockRef<uint32_t>::dense(pool->intern(std::move(col))));
        }
        const BlockMat<uint32_t> A = b.finalize();
        blk::BlockRrefResult<uint32_t> ref;
        { tp::ScopedThreadDevice cpu(tp::ThreadDevice::Cpu); ref = blk::rref<uint32_t>(A); }
        tp::ScopedThreadDevice gpu(tp::ThreadDevice::Gpu);
        const blk::BlockRrefResult<uint32_t> got = tp::rref<uint32_t>(ex, A);
        check(got.pivot_rows == ref.pivot_rows && got.pivots == ref.pivots && same_entries(cview(got.R), cview(ref.R)),
              "(B) p=" + str(p) + " one-column panels: pivot_rows in pivot order, pivots, R");
        check(got.pivot_rows == rule_reference(ctx, M), "(B) p=" + str(p) + " one-column panels: == the smallest-row rule");
    }
    {   // the host re-run of ENGINE_VERIFY_DEVICE builds, armed at run time: it runs (counter) and agrees (no throw)
        ScopedEnv ver("ENGINE_GPU_ELIM_VERIFY", "1");
        const uint64_t v0 = tp::device_elim_totals().verified;
        auto pool = std::make_shared<BlockPool<uint32_t>>();
        const BlockMat<uint32_t> A = tp::test::make_input<uint32_t>(rng, pool, 13u, tp::test::Shape::Generic, std::vector<uint32_t>(6, 65),
                                                                    std::vector<uint32_t>(7, 65), 40);
        Buf D = A.to_dense();
        bool ok = true;
        try {
            tp::ScopedThreadDevice gpu(tp::ThreadDevice::Gpu);
            (void)tp::rref<uint32_t>(ex, A);
            (void)elim::rref<uint32_t>(ModCtx<uint32_t>(13), view(D));
            Buf D2 = A.to_dense();
            (void)elim::column_rank_profile<uint32_t>(ModCtx<uint32_t>(13), view(D2));
        } catch (const std::exception& e) { ok = false; std::cout << "  verify: " << e.what() << std::endl; }
        check(ok && tp::device_elim_totals().verified - v0 == 3, "(B) ENGINE_GPU_ELIM_VERIFY=1: 3 device eliminations re-run on the host and equal");
    }
    std::cout << "  (B) NC: the device's raw pivot order differs from the driver's per-panel order on " << order_differs << " of "
              << order_cases << " blocked cases" << std::endl;
    check(order_differs > 0, "(B) NC: the per-panel reordering of pivot_rows is exercised (raw order differs somewhere)");
}

// ---- (C) the DenseElim hook ---------------------------------------------------------------
void arm_c() {
    ScopedEnv min0("ENGINE_GPU_ELIM_MIN", "0");
    Rng rng(0xC0C);
    for (uint32_t p : {2u, 13u, 127u}) {
        const ModCtx<uint32_t> ctx(p);
        const std::string tag = "(C) p=" + str(p);
        const tp::DeviceElimTotals t0 = tp::device_elim_totals();
        for (Layout lay : {Layout::RowMajor, Layout::ColMajor}) {
            const Buf A0 = make_matrix(rng, 210, 260, p, Kind::Boundaries);
            Buf A(210, 260, lay);
            for (uint32_t i = 0; i < 210; ++i) for (uint32_t j = 0; j < 260; ++j) A.at(i, j) = A0.at(i, j);
            Buf B(A);
            elim::RrefResult rg, rc;
            { tp::ScopedThreadDevice gpu(tp::ThreadDevice::Gpu); rg = elim::rref<uint32_t>(ctx, view(A)); }
            { tp::ScopedThreadDevice cpu(tp::ThreadDevice::Cpu); rc = elim::rref<uint32_t>(ctx, view(B)); }
            check(rg.rank == rc.rank && rg.pivots == rc.pivots && same_entries(cview(A), cview(B)),
                  tag + std::string(" elim::rref in place, ") + (lay == Layout::RowMajor ? "row" : "col") + "-major");
        }
        {
            const Buf A0 = make_matrix(rng, 230, 190, p, Kind::DupRows);
            Buf A(A0), B(A0);
            elim::RrefResult rg, rc;
            { tp::ScopedThreadDevice gpu(tp::ThreadDevice::Gpu); rg = elim::column_rank_profile<uint32_t>(ctx, view(A)); }
            { tp::ScopedThreadDevice cpu(tp::ThreadDevice::Cpu); rc = elim::column_rank_profile<uint32_t>(ctx, view(B)); }
            check(rg.rank == rc.rank && rg.pivots == rc.pivots, tag + " column_rank_profile");
        }
        {
            Buf A = make_matrix(rng, 150, 150, p, Kind::Random);
            Buf Ig(150, 150), Ic(150, 150);
            bool okg, okc;
            { tp::ScopedThreadDevice gpu(tp::ThreadDevice::Gpu); okg = elim::inverse<uint32_t>(ctx, cview(A), view(Ig)); }
            { tp::ScopedThreadDevice cpu(tp::ThreadDevice::Cpu); okc = elim::inverse<uint32_t>(ctx, cview(A), view(Ic)); }
            check(okg == okc && (!okc || same_entries(cview(Ig), cview(Ic))), tag + " inverse");
            Buf S = make_matrix(rng, 150, 150, p, Kind::Planted, 120);
            { tp::ScopedThreadDevice gpu(tp::ThreadDevice::Gpu); okg = elim::inverse<uint32_t>(ctx, cview(S), view(Ig)); }
            check(!okg, tag + " inverse refuses a singular matrix");
        }
        {
            Buf Bm = make_matrix(rng, 180, 60, p, Kind::Planted, 50), coef = make_matrix(rng, 60, 7, p, Kind::Random), Y(180, 7);
            kernels::gemm<uint32_t>(ctx, cview(Bm), cview(coef), view(Y), kernels::GemmOp::Set);
            std::vector<std::vector<uint32_t>> xg, xc;
            { tp::ScopedThreadDevice gpu(tp::ThreadDevice::Gpu); xg = elim::solve_multi_rhs<uint32_t>(ctx, cview(Bm), cview(Y)); }
            { tp::ScopedThreadDevice cpu(tp::ThreadDevice::Cpu); xc = elim::solve_multi_rhs<uint32_t>(ctx, cview(Bm), cview(Y)); }
            check(xg == xc, tag + " solve_multi_rhs");
            Buf Yo(Y);
            for (uint32_t i = 0; i < 180; ++i) Yo.at(i, 2) = rng.elt<uint32_t>(p);
            std::string m1, m2;
            { tp::ScopedThreadDevice gpu(tp::ThreadDevice::Gpu); try { (void)elim::solve_multi_rhs<uint32_t>(ctx, cview(Bm), cview(Yo)); } catch (const std::runtime_error& e) { m1 = e.what(); } }
            { tp::ScopedThreadDevice cpu(tp::ThreadDevice::Cpu); try { (void)elim::solve_multi_rhs<uint32_t>(ctx, cview(Bm), cview(Yo)); } catch (const std::runtime_error& e) { m2 = e.what(); } }
            check(!m1.empty() && m1 == m2, tag + " solve_multi_rhs out-of-span message");
        }
        {
            Buf Rl = make_matrix(rng, 160, 20, p, Kind::Planted, 15), G = make_matrix(rng, 160, 50, p, Kind::Random);
            for (uint32_t i = 0; i < 160; ++i) G.at(i, 4) = Rl.at(i, 2);
            Buf Pg, Pc;
            std::vector<uint32_t> gg, gc;
            { tp::ScopedThreadDevice gpu(tp::ThreadDevice::Gpu); gg = elim::complement_basis<uint32_t>(ctx, cview(Rl), cview(G), &Pg); }
            { tp::ScopedThreadDevice cpu(tp::ThreadDevice::Cpu); gc = elim::complement_basis<uint32_t>(ctx, cview(Rl), cview(G), &Pc); }
            check(gg == gc && same_entries(cview(Pg), cview(Pc)), tag + " complement_basis (projector)");
        }
        {
            Buf A = make_matrix(rng, 120, 170, p, Kind::Boundaries);
            Buf Rg(120, 170), Ug(120, 120), Rc(120, 170), Uc(120, 120);
            elim::RrefResult a1, a2;
            { tp::ScopedThreadDevice gpu(tp::ThreadDevice::Gpu); a1 = elim::rref_with_transform<uint32_t>(ctx, cview(A), view(Rg), view(Ug)); }
            { tp::ScopedThreadDevice cpu(tp::ThreadDevice::Cpu); a2 = elim::rref_with_transform<uint32_t>(ctx, cview(A), view(Rc), view(Uc)); }
            check(a1.pivots == a2.pivots && same_entries(cview(Rg), cview(Rc)) && same_entries(cview(Ug), cview(Uc)), tag + " rref_with_transform");
        }
        const tp::DeviceElimTotals t1 = tp::device_elim_totals();
        // rref x2, inverse x2, solve x2, complement, rref_with_transform: 8 in-place RREFs; 1 rank profile
        check(t1.dense_rrefs - t0.dense_rrefs == 8 && t1.rank_profiles - t0.rank_profiles == 1,
              tag + ": the hook ran on the device " + str(t1.dense_rrefs - t0.dense_rrefs) + " + " + str(t1.rank_profiles - t0.rank_profiles) + " times (expected 8 + 1)");
    }
}

// ---- (D) routing negatives ----------------------------------------------------------------
void arm_d() {
    ScopedEnv min0("ENGINE_GPU_ELIM_MIN", "0");
    Rng rng(0xD0D);
    const ModCtx<uint32_t> c13(13);
    const uint64_t calls0 = cu::imma_elim_stats().calls;
    auto same_counters = [&](const std::string& what) {
        check(cu::imma_elim_stats().calls == calls0, "(D) " + what + ": imma_rref not called");
    };
    // (S24 CL-R2) ENGINE_GPU_ELIM is a RETIRED lever: its "0" arm is a legacy arm (an OFF build refuses "0")
    if (oc::legacy::RunLegacyArm("CudaImmaElimValidation", "D-gpu-elim-off")) {
        ScopedEnv off("ENGINE_GPU_ELIM", "0");
        tp::ScopedThreadDevice gpu(tp::ThreadDevice::Gpu);
        Buf A = make_matrix(rng, 200, 230, 13, Kind::Random), B(A);
        const elim::RrefResult a = elim::rref<uint32_t>(c13, view(A));
        const elim::RrefResult b = cpu_rref(c13, B);
        check(a.pivots == b.pivots && same_entries(cview(A), cview(B)), "(D) ENGINE_GPU_ELIM=0: same values");
        check(!tp::device_elim_wanted(c13, 200, 230), "(D) ENGINE_GPU_ELIM=0: not wanted");
        same_counters("ENGINE_GPU_ELIM=0");
    }
    {
        tp::ScopedThreadDevice cpu(tp::ThreadDevice::Cpu);
        check(!tp::device_elim_wanted(c13, 200, 230), "(D) Cpu override: not wanted");
        Buf A = make_matrix(rng, 200, 230, 13, Kind::Random);
        (void)elim::rref<uint32_t>(c13, view(A));
        same_counters("Cpu override");
    }
    {
        tp::ScopedThreadDevice gpu(tp::ThreadDevice::Gpu);
        check(!tp::device_elim_wanted(ModCtx<uint32_t>(131), 200, 230), "(D) p = 131: not wanted");
        check(!tp::device_elim_wanted(ModCtx<uint32_t>(134217757u), 200, 230), "(D) p = 134217757: not wanted");
        check(!tp::device_elim_wanted(ModCtx<uint32_t>::prime_power(3u, 2), 200, 230), "(D) Z/9: not wanted (field-only route)");
        check(!cu::imma_elim_eligible(ModCtx<uint32_t>::prime_power(5u, 3)), "(D) Z/125: not eligible");
        Buf A = make_matrix(rng, 200, 230, 131, Kind::Random);
        (void)elim::rref<uint32_t>(ModCtx<uint32_t>(131), view(A));
        same_counters("p = 131");
        bool threw = false;
        try { (void)cu::imma_rref(ModCtx<uint32_t>(131), 2, 2, [&](uint32_t, uint32_t) { return cview(A).sub(0, 2, 0, 2); },
                                  cu::ImmaRowSink{}); } catch (const std::logic_error&) { threw = true; }
        check(threw, "(D) imma_rref refuses p = 131");
        {
            ScopedEnv big("ENGINE_GPU_ELIM_MIN", "46001");
            check(!tp::device_elim_wanted(c13, 200, 230) && tp::device_elim_wanted(c13, 200, 231), "(D) the size floor ENGINE_GPU_ELIM_MIN (46000 < 46001 <= 46200)");
            Buf B = make_matrix(rng, 200, 230, 13, Kind::Random);
            (void)elim::rref<uint32_t>(c13, view(B));
            same_counters("below ENGINE_GPU_ELIM_MIN");
        }
        tp::Executor<uint64_t> ex64;
        auto pool = std::make_shared<BlockPool<uint64_t>>();
        const BlockMat<uint64_t> A64 = tp::test::make_input<uint64_t>(rng, pool, 13ull, tp::test::Shape::Generic, {9, 9}, {9, 9, 9}, 60);
        const blk::BlockRrefResult<uint64_t> g64 = tp::rref<uint64_t>(ex64, A64);
        blk::BlockRrefResult<uint64_t> r64;
        { tp::ScopedThreadDevice cpu(tp::ThreadDevice::Cpu); r64 = blk::rref<uint64_t>(A64); }
        check(g64.pivots == r64.pivots, "(D) uint64_t context: values");
        same_counters("uint64_t tape::rref");
    }
}

// ---- (E) launch counts ------------------------------------------------------------------
void arm_e() {
    Rng rng(0xE0E);
    const ModCtx<uint32_t> ctx(13);
    for (uint32_t bw : {32u, 16u}) {
        const uint32_t m = 1100, n = 1000;   // full column rank (random, m > n): every base finds W pivots
        const Buf A = make_matrix(rng, m, n, 13, Kind::Random);
        const cu::ImmaElimStats s0 = cu::imma_elim_stats();
        const uint64_t il0 = cu::launch_stats().imma_launches;
        const DirectOut d = device_rref(ctx, A, 0, bw);
        const cu::ImmaElimStats s1 = cu::imma_elim_stats();
        const uint64_t il1 = cu::launch_stats().imma_launches;
        const uint64_t bases = s1.bases - s0.bases, coop = s1.coop_launches - s0.coop_launches, gemms = s1.gemms - s0.gemms;
        check(d.r.rank == n, "(E) W=" + str(bw) + ": full column rank");
        check(bases == (n + bw - 1) / bw, "(E) W=" + str(bw) + ": bases " + str(bases) + " == ceil(n / W) = " + str((n + bw - 1) / bw));
        check(coop == bases, "(E) W=" + str(bw) + ": one cooperative launch per base");
        check(s1.base_columns - s0.base_columns == n, "(E) W=" + str(bw) + ": base columns == n");
        // inner updates: one U12 + one Schur GEMM per base except the last of each outer panel; outer: 2 per panel
        // except the last; to RREF: 1 per 256 pivot rows + 1 for each block with rows above. Asserted >= the outer count.
        check(gemms >= 2 * ((n + 255) / 256 - 1) + (n + 255) / 256, "(E) W=" + str(bw) + ": GEMMs " + str(gemms));
        check(il1 - il0 == gemms, "(E) W=" + str(bw) + ": IMMA launches " + str(il1 - il0) + " == the elimination's GEMMs");
        std::cout << "  (E) 1100x1000 p=13 W=" << bw << ": bases " << bases << ", cooperative launches " << coop << ", IMMA GEMMs "
                  << gemms << ", kernel launches " << (s1.kernel_launches - s0.kernel_launches) << std::endl;
    }
    {   // a zero matrix: every base runs, no GEMM at all
        const Buf Z(300, 300);
        const cu::ImmaElimStats s0 = cu::imma_elim_stats();
        const DirectOut d = device_rref(ctx, Z);
        const cu::ImmaElimStats s1 = cu::imma_elim_stats();
        check(d.r.rank == 0 && s1.bases - s0.bases == (300 + 31) / 32 && s1.gemms == s0.gemms, "(E) zero matrix: rank 0, ceil(n/32) bases, 0 GEMMs");
    }
}

// ---- (F) negative controls ----------------------------------------------------------------
void arm_f() {
    Rng rng(0xF0F);
    const ModCtx<uint32_t> ctx(13);
    const Buf A = make_matrix(rng, 400, 380, 13, Kind::DupRows);
    Buf Ap(A);
    Ap.at(123, 77) = ctx.add(Ap.at(123, 77), 1u);
    Buf ref(A);
    const elim::RrefResult er = cpu_rref(ctx, ref);
    const DirectOut d = device_rref(ctx, Ap);
    bool same = d.r.rank == er.rank && d.r.pivots == er.pivots;
    for (uint32_t i = 0; same && i < er.rank; ++i) same = std::memcmp(&d.R.at(i, 0), &ref.at(i, 0), sizeof(uint32_t) * 380) == 0;
    check(!same, "(F) NC: a one-entry perturbation of the input is detected by the comparison");
    const DirectOut d0 = device_rref(ctx, A);
    check(d0.r.pivot_rows != rule_reference(ctx, A, true), "(F) NC: the largest-row rule disagrees with pivot_rows");
    check(d0.r.pivot_rows == rule_reference(ctx, A, false), "(F) the smallest-row rule agrees (control of the NC)");
}

// ---- (G) optional timing arm (read-only hook CUDA_IMMA_ELIM_BENCH="m,n[,p]"; CUDA_IMMA_ELIM_BENCH_CPU=1 adds the host) ----
void arm_g() {
    const char* e = std::getenv("CUDA_IMMA_ELIM_BENCH");
    if (e == nullptr || e[0] == '\0') return;
    unsigned m = 0, n = 0, p = 13;
    if (std::sscanf(e, "%u,%u,%u", &m, &n, &p) < 2 || m == 0 || n == 0 || !cu::imma_elim_eligible(ModCtx<uint32_t>(p))) {
        std::cout << "  (G) REFUSED CUDA_IMMA_ELIM_BENCH=" << e << " (expected m,n[,p] with p <= 127 prime)" << std::endl;
        ++g_fail;
        return;
    }
    const bool cpu = std::getenv("CUDA_IMMA_ELIM_BENCH_CPU") != nullptr;
    std::cout << "EFFECTIVE CudaImmaElimValidation arm=bench m=" << m << " n=" << n << " p=" << p << " cpu=" << (cpu ? 1 : 0) << std::endl;
    Rng rng(0x6E6E);
    const ModCtx<uint32_t> ctx(p);
    for (Kind kind : {Kind::Random, Kind::Sparse}) {
        const Buf A = make_matrix(rng, m, n, p, kind);
        double best = 1e30;
        DirectOut d;
        for (int rep = 0; rep < 3; ++rep) {
            const auto t0 = std::chrono::steady_clock::now();
            d = device_rref(ctx, A);
            best = std::min(best, std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
        }
        std::cout << "  (G) " << m << "x" << n << " p=" << p << " " << kname(kind) << ": rank " << d.r.rank << ", device (incl. transfers) "
                  << best << " s, " << cu::imma_rref_device_bytes(m, n) / 1048576.0 << " MB" << std::flush;
        if (cpu) {
            Buf ref(A);
            const auto t0 = std::chrono::steady_clock::now();
            const elim::RrefResult er = cpu_rref(ctx, ref);
            const double tc = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
            bool same = d.r.rank == er.rank && d.r.pivots == er.pivots;
            for (uint32_t i = 0; same && i < er.rank; ++i) same = std::memcmp(&d.R.at(i, 0), &ref.at(i, 0), sizeof(uint32_t) * n) == 0;
            check(same, "(G) bench case identical to the host");
            std::cout << "; host elim::rref (1 thread) " << tc << " s, identical " << (same ? "yes" : "NO");
        }
        std::cout << std::endl;
    }
}

// ---- (H) the fallback is loud (audit R1) -------------------------------------------------------
// ENGINE_GPU_ELIM_SIMULATE_CUDA_ERROR=1 (test-only) raises a CudaError where an OOM / refused cooperative launch would: the
// host route must run and return the same values, `fallbacks` must move, one stderr line per fallback, and the
// `EFFECTIVE DEVICE_ELIM` summary line must report it.
void arm_h() {
    ScopedEnv min0("ENGINE_GPU_ELIM_MIN", "0");
    Rng rng(0x4A4);
    const ModCtx<uint32_t> c13(13);
    const tp::DeviceElimTotals t0 = tp::device_elim_totals();
    std::ostringstream captured;
    std::streambuf* old = std::cerr.rdbuf(captured.rdbuf());
    bool same = true;
    try {
        ScopedEnv sim("ENGINE_GPU_ELIM_SIMULATE_CUDA_ERROR", "1");
        tp::ScopedThreadDevice gpu(tp::ThreadDevice::Gpu);
        Buf A = make_matrix(rng, 120, 140, 13, Kind::Random), B(A);
        const elim::RrefResult a = elim::rref<uint32_t>(c13, view(A));
        { tp::ScopedThreadDevice cpu(tp::ThreadDevice::Cpu); const elim::RrefResult b = elim::rref<uint32_t>(c13, view(B));
          same = a.rank == b.rank && a.pivots == b.pivots && same_entries(cview(A), cview(B)); }
    } catch (...) { std::cerr.rdbuf(old); throw; }
    std::cerr.rdbuf(old);
    const tp::DeviceElimTotals t1 = tp::device_elim_totals();
    check(same, "(H) simulated CudaError: the host fallback returns the same RREF");
    check(t1.fallbacks - t0.fallbacks == 1, "(H) simulated CudaError: fallbacks incremented by exactly 1");
    check(t1.dense_rrefs == t0.dense_rrefs, "(H) simulated CudaError: no device elimination counted");
    check(captured.str().find("[imma_rref] fallback to the host route (DenseElim, 120x140)") != std::string::npos,
          "(H) one stderr line per fallback, with route and shape: '" + captured.str() + "'");
    const std::string line = tp::device_elim_summary_line();
    std::cout << "  " << line << std::endl;
    check(line.rfind("EFFECTIVE DEVICE_ELIM calls=", 0) == 0 && line.find(" fallbacks=" + str(t1.fallbacks)) != std::string::npos,
          "(H) the EFFECTIVE DEVICE_ELIM summary line reports the fallback count");
}

} // namespace

uint32_t CudaImmaElimValidation() {
    std::cout << "=== CudaImmaElimValidation (device-resident IMMA elimination vs the host engine, entrywise) ===" << std::endl;
    std::cout << engine::lever::retired_device_lever_line(engine::lever::gpu_elim_lever()) << std::endl;   // (S24 CL-R2)
    if (!cu::available()) {
        std::cout << "[SKIP] CudaImmaElimValidation: no usable CUDA device (" << cu::unavailable_reason()
                  << ") -- nothing to validate, returning 0" << std::endl;
        return 0;
    }
    if (!cu::imma_supported()) {
        std::cout << "[SKIP] CudaImmaElimValidation: the device / build has no sm_80+ IMMA image -- the device elimination is never "
                     "taken, nothing to validate, returning 0" << std::endl;
        return 0;
    }
    g_checks = 0; g_fail = 0;
    const auto t0 = std::chrono::steady_clock::now();
    const cu::ImmaElimStats s0 = cu::imma_elim_stats();
    ScopedEnv on("ENGINE_GPU_ELIM", nullptr);   // the default (on) for arms A-C, E, F; D switches it itself
    try {
        const bool bench_only = std::getenv("CUDA_IMMA_ELIM_BENCH_ONLY") != nullptr;   // the timing arm alone (profiling runs)
        if (bench_only) { arm_g(); throw std::runtime_error("CUDA_IMMA_ELIM_BENCH_ONLY: arms A-F skipped (a timing run is never a pass)"); }
        arm_a(); std::cout << "  (A) done: " << g_checks << " checks" << std::endl;
        arm_b(); std::cout << "  (B) done: " << g_checks << " checks" << std::endl;
        arm_c(); std::cout << "  (C) done: " << g_checks << " checks" << std::endl;
        arm_d(); std::cout << "  (D) done: " << g_checks << " checks" << std::endl;
        arm_e(); std::cout << "  (E) done: " << g_checks << " checks" << std::endl;
        arm_f(); std::cout << "  (F) done: " << g_checks << " checks" << std::endl;
        arm_h(); std::cout << "  (H) done: " << g_checks << " checks" << std::endl;
        arm_g();
    } catch (const std::exception& e) {
        std::cout << "  [EXCEPTION] " << e.what() << std::endl;
        ++g_fail;
    }
    const cu::ImmaElimStats s1 = cu::imma_elim_stats();
    const double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    std::cout << "CUDA_IMMA_ELIM checks=" << g_checks << " failing=" << g_fail << " device_calls=" << (s1.calls - s0.calls)
              << " bases=" << (s1.bases - s0.bases) << " gemms=" << (s1.gemms - s0.gemms) << " wall_s=" << secs << std::endl;
    std::cout << "CudaImmaElimValidation: " << g_checks << " checks, " << g_fail << " failures" << std::endl;
    return g_fail == 0 ? 0 : 1;
}
