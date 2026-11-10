// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// EngineArenaValidation.cpp -- the arena allocator: lifetimes in wavefront
// levels never overlap in memory, arena results == per-Temp owned storage, a
// poisoned arena changes nothing (no node reads what it did not write), the
// arena is allocated once and reused across replays, and sharing actually
// happens.  Run on the Check build (ASan/UBSan) for the use-after-free /
// out-of-bounds claim.
#include "EngineArenaValidation.h"

#include <iostream>

#include "../ModArithmetic/BlockKernels.h"
#include "EngineTestUtil.h"
#include "Executor.h"
#include "Passes.h"
#include "Recorder.h"
#include "TapeElim.h"

namespace {

using namespace engine;
using namespace engine::tape;
using namespace engine::tape::test;
using kernels::GemmOp;

Checker g;

template<class Elt>
void check_lifetimes(const Program<Elt>& prog, const std::string& tag) {
    uint32_t live = 0;
    size_t sum = 0;
    for (BufId x = 0; x < prog.bufs.size(); ++x) {
        const BufferDesc& a = prog.bufs[x];
        if (a.kind != BufKind::Temp || a.footprint == 0) continue;
        ++live; sum += a.footprint;
        g.check(a.arena_offset + a.footprint <= prog.arena_elts, tag + " temp inside the arena");
        g.check(a.arena_offset % 16 == 0, tag + " temp 64-byte aligned");
        for (BufId y = x + 1; y < prog.bufs.size(); ++y) {
            const BufferDesc& b = prog.bufs[y];
            if (b.kind != BufKind::Temp || b.footprint == 0) continue;
            const bool levels_overlap = a.first_level <= b.last_level && b.first_level <= a.last_level;
            const bool mem_overlap = a.arena_offset < b.arena_offset + b.footprint && b.arena_offset < a.arena_offset + a.footprint;
            if (levels_overlap && mem_overlap) g.check(false, tag + " two live temps share arena memory");
        }
    }
    if (live > 0) g.check(prog.arena_elts <= sum + 16 * live, tag + " arena <= sum of footprints (+alignment)");
    // every referenced temp has a lifetime, every unreferenced one has footprint 0
    std::vector<Access> acc;
    std::vector<char> referenced(prog.bufs.size(), 0);
    for (const Node<Elt>& n : prog.nodes) {
        if (n.dead) continue;
        node_accesses(prog, n, acc);
        for (const Access& x : acc) referenced[x.v.buf] = 1;
    }
    for (BufId x = 0; x < prog.bufs.size(); ++x) {
        const BufferDesc& d = prog.bufs[x];
        if (d.kind != BufKind::Temp) continue;
        if (!referenced[x]) g.check(d.footprint == 0, tag + " unreferenced temp has no footprint");
        else if (d.rows > 0 && d.cols > 0) g.check(d.footprint > 0 && d.first_level <= d.last_level, tag + " referenced temp has a lifetime");
    }
}

// A program with a long chain of temps (sequential products) so that the
// arena must reuse memory: Y = A_k ... A_1 X block-wise, then a Theta-like
// accumulation on top.
template<class Elt>
void chain_case(Elt p, uint64_t seed, const std::string& tag) {
    const ModCtx<Elt> ctx(p);
    auto pool = std::make_shared<BlockPool<Elt>>();
    Rng rng(seed);
    const std::vector<uint32_t> dims(5, 33);
    const BlockMat<Elt> A = make_input<Elt>(rng, pool, p, Shape::Generic, dims, dims, 50, 4);
    const BlockMat<Elt> B = make_input<Elt>(rng, pool, p, Shape::CqLike, dims, dims, 50, 4);
    const uint32_t w = 9, n = A.cols();
    const DenseBuf<Elt> X = random_buf<Elt>(rng, n, w, p, Layout::ColMajor);
    // reference: T1 = A X; T2 = B T1; T3 = A T2; Y = T3 + B X
    DenseBuf<Elt> T1(n, w), T2(n, w), T3(n, w), Yref(n, w, Layout::ColMajor);
    blk::gemm_dense<Elt>(A, cview(X), view(T1), GemmOp::Set);
    blk::gemm_dense<Elt>(B, cview(T1), view(T2), GemmOp::Set);
    blk::gemm_dense<Elt>(A, cview(T2), view(T3), GemmOp::Set);
    kernels::copy<Elt>(cview(T3), view(Yref));
    blk::gemm_dense<Elt>(B, cview(X), view(Yref), GemmOp::Add);
    auto record = [&](DenseBuf<Elt>& Y) {
        Recorder<Elt> rec(ctx);
        const BufId bX = rec.input(cview(X)), bY = rec.output(view(Y));
        const BufId t1 = rec.temp(n, w), t2 = rec.temp(n, w, Layout::ColMajor), t3 = rec.temp(n, w);
        rec.record_gemm_dense(A, rec.whole(bX), rec.whole(t1), GemmOp::Set);
        rec.record_gemm_dense(B, rec.whole(t1), rec.whole(t2), GemmOp::Set);
        rec.record_gemm_dense(A, rec.whole(t2), rec.whole(t3), GemmOp::Set);
        rec.scatter(rec.whole(t3), rec.whole(bY));
        rec.record_gemm_dense(B, rec.whole(bX), rec.whole(bY), GemmOp::Add);
        return rec;
    };
    Executor<Elt> ex;
    // arena vs owned storage vs poisoned arena
    for (int mode = 0; mode < 3; ++mode) {
        DenseBuf<Elt> Y(n, w, Layout::ColMajor);
        Recorder<Elt> rec = record(Y);
        Program<Elt> prog = rec.finish();
        PassOptions o; o.arena = (mode != 1);
        ex.set_poison(mode == 2);
        ex.run(prog, rec.bindings(), o);
        g.same(cview(Y), cview(Yref), tag + (mode == 0 ? " arena" : mode == 1 ? " owned storage" : " poisoned arena"));
        if (mode != 1) check_lifetimes(prog, tag);
        if (mode == 0) {
            g.check(prog.stats.temp_elts > prog.arena_elts, tag + " arena reuses memory (arena < sum of temps)");
            g.check(prog.levels.size() >= 4, tag + " chain has >= 4 wavefronts");
        }
    }
    ex.set_poison(false);
    // replay: the arena is allocated once and reused (same base pointer, same capacity)
    {
        DenseBuf<Elt> Y(n, w, Layout::ColMajor);
        Recorder<Elt> rec = record(Y);
        Program<Elt> prog = rec.finish();
        ex.run(prog, rec.bindings(), PassOptions());
        const size_t cap = ex.arena_capacity();
        const Elt* base = ex.arena_base();
        for (int rep = 0; rep < 5; ++rep) {
            DenseBuf<Elt> Y2(n, w, Layout::ColMajor);
            Bindings<Elt> b = rec.bindings(); b.outputs[0] = view(Y2);
            ex.run(prog, b);
            g.same(cview(Y2), cview(Yref), tag + " replay " + std::to_string(rep));
        }
        g.check(ex.arena_capacity() == cap && ex.arena_base() == base, tag + " arena allocated once, reused across replays");
    }
}

template<class Elt>
void elim_case(Elt p, uint64_t seed, const std::string& tag) {
    auto pool = std::make_shared<BlockPool<Elt>>();
    Rng rng(seed);
    const BlockMat<Elt> A = make_input<Elt>(rng, pool, p, Shape::CqLike, std::vector<uint32_t>(6, 65), std::vector<uint32_t>(7, 65), 25, 6);
    const blk::BlockRrefResult<Elt> ref = blk::rref<Elt>(A);
    Executor<Elt> ex;
    ex.set_poison(true);
    const blk::BlockRrefResult<Elt> got = rref<Elt>(ex, A);
    ex.set_poison(false);
    g.check(got.pivots == ref.pivots, tag + " elimination pivots under a poisoned arena");
    g.same(cview(got.R), cview(ref.R), tag + " elimination R under a poisoned arena");
}

} // namespace

uint32_t EngineArenaValidation() {
    const auto t0 = std::chrono::steady_clock::now();
    g = Checker();
    std::cout << "=== EngineArenaValidation (lifetimes, arena vs owned storage, poisoned arena, replay reuse) ===" << std::endl;
    try {
        chain_case<uint32_t>(134217757u, 0xA7E1, "[u32 p=134217757]");
        chain_case<uint32_t>(4294967291u, 0xA7E2, "[u32 p=2^32-5]");
        chain_case<uint64_t>(2305843009213693951ull, 0xA7E3, "[u64 p=2^61-1]");
        elim_case<uint32_t>(134217757u, 0xA7E4, "[u32 p=134217757]");
        elim_case<uint64_t>(134217757ull, 0xA7E5, "[u64 p=134217757]");
    } catch (const std::exception& e) {
        std::cout << "  EXCEPTION: " << e.what() << std::endl;
        ++g.fails;
    }
    std::cout << "EngineArenaValidation: " << g.checks << " checks, " << g.fails << " failed, " << seconds_since(t0) << " s" << std::endl;
    return g.fails == 0 ? 0 : 1;
}
