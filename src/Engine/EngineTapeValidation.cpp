// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// EngineTapeValidation.cpp -- record -> passes -> execute equals direct
// execution (matrix-engine Phase 4, plan D4 "EngineTapeValidation").
//
// Programs shaped like the pipeline's: a Theta-like accumulation of many
// same-shape block GEMMs with shared pool ids (record_gemm_dense against
// blk::gemm_dense), a block x block product (against blk::gemm), a
// GATHER / SCATTER round trip (against to_dense), and the op set on its own
// (AXPY incl. the identity source, SCAL, TRSM kinds, CANONICALIZE, COMPARE).
// Every pass is switched off / on / forced (CSE, fold, fusion, layout,
// k-blocking, batching, arena, the device model) and must give identical
// buffers; a cached program replayed on new data of the same shape equals a
// fresh recording, and a changed pattern hash MUST miss.
#include "EngineTapeValidation.h"

#include <iostream>
#include <map>

#include "../ModArithmetic/BlockKernels.h"
#include "../ModArithmetic/Kernels.h"
#include "EngineTestUtil.h"
#include "Executor.h"
#include "Passes.h"
#include "ProgramCache.h"
#include "Recorder.h"

namespace {

using namespace engine;
using namespace engine::tape;
using namespace engine::tape::test;
using kernels::GemmOp;

Checker g;
void trace(const std::string& what) { if (std::getenv("ENGINE_TAPE_TRACE")) std::cout << "  -- " << what << std::endl; }

std::vector<std::pair<std::string, PassOptions>> option_arms() {
    std::vector<std::pair<std::string, PassOptions>> arms;
    PassOptions d;
    arms.emplace_back("default", d);
    { PassOptions o = d; o.cse = false; arms.emplace_back("cse=0", o); }
    { PassOptions o = d; o.fold = false; arms.emplace_back("fold=0", o); }
    { PassOptions o = d; o.fuse = false; arms.emplace_back("fuse=0", o); }
    { PassOptions o = d; o.layout = false; arms.emplace_back("layout=0", o); }
    { PassOptions o = d; o.ksplit = true; arms.emplace_back("ksplit=1", o); }
    { PassOptions o = d; o.batch = false; arms.emplace_back("batch=0", o); }
    { PassOptions o = d; o.arena = false; arms.emplace_back("arena=0", o); }
    { PassOptions o = d; o.device_model = true; o.device_policy = Device::GPU; arms.emplace_back("device=hybrid", o); }
    { PassOptions o = d; o.policy = DevicePolicy::Gpu; arms.emplace_back("device=gpu", o); }   // Phase 5: on the device when one exists
    { PassOptions o = d; o.cse = false; o.fuse = false; o.batch = false; o.layout = false; arms.emplace_back("all-off", o); }
    return arms;
}

template<class Elt>
void theta_like(Executor<Elt>& ex, Elt p, const std::vector<uint32_t>& rd, const std::vector<uint32_t>& cd,
                Shape shape, uint32_t density, uint32_t w, uint64_t seed, const std::string& tag) {
    const ModCtx<Elt> ctx(p);
    auto pool = std::make_shared<BlockPool<Elt>>();
    Rng rng(seed);
    const BlockMat<Elt> A = make_input<Elt>(rng, pool, p, shape, rd, cd, density, 8);
    trace(tag + " theta_like");
    for (Layout xl : {Layout::ColMajor, Layout::RowMajor}) {
        const DenseBuf<Elt> X = random_buf<Elt>(rng, A.cols(), w, p, xl);
        for (GemmOp gop : {GemmOp::Set, GemmOp::Add, GemmOp::Sub}) {
            const DenseBuf<Elt> Y0 = random_buf<Elt>(rng, A.rows(), w, p, Layout::ColMajor);
            DenseBuf<Elt> Yref(Y0);
            blk::gemm_dense<Elt>(A, cview(X), view(Yref), gop);
            for (const auto& arm : option_arms()) {
                for (Layout yl : {Layout::ColMajor, Layout::RowMajor}) {
                    DenseBuf<Elt> Y(A.rows(), w, yl);
                    kernels::copy<Elt>(cview(Y0), view(Y));
                    Recorder<Elt> rec(ctx);
                    const BufId bX = rec.input(cview(X));
                    const BufId bY = rec.output(view(Y));
                    rec.record_gemm_dense(A, rec.whole(bX), rec.whole(bY), gop);
                    Program<Elt> prog = rec.finish();
                    ex.run(prog, rec.bindings(), arm.second);
                    g.same(cview(Y), cview(Yref), tag + " theta-like gemm_dense " + arm.first +
                           " X" + (xl == Layout::ColMajor ? "col" : "row") + " Y" + (yl == Layout::ColMajor ? "col" : "row") +
                           " gop=" + std::to_string(static_cast<int>(gop)));
                    if (arm.first == "default" && gop == GemmOp::Set && xl == Layout::ColMajor && yl == Layout::ColMajor) {
                        g.check(prog.stats.transposes_inserted <= 1, tag + " one relayout of the column-major multi-vector");
                        if (std::getenv("ENGINE_TAPE_TRACE")) std::cout << "    " << tag << ": " << prog.summary() << std::endl;
                    }
                }
            }
        }
    }
}

template<class Elt>
void block_product(Executor<Elt>& ex, Elt p, const std::vector<uint32_t>& rd, const std::vector<uint32_t>& kd,
                   const std::vector<uint32_t>& cd, uint64_t seed, const std::string& tag) {
    const ModCtx<Elt> ctx(p);
    auto pool = std::make_shared<BlockPool<Elt>>();
    Rng rng(seed);
    const BlockMat<Elt> A = make_input<Elt>(rng, pool, p, Shape::Generic, rd, kd, 60, 6);
    const BlockMat<Elt> B = make_input<Elt>(rng, pool, p, Shape::Generic, kd, cd, 60, 6);
    const DenseBuf<Elt> Cref = blk::gemm<Elt>(A, B).to_dense(Layout::RowMajor);
    trace(tag + " block_product");
    for (const auto& arm : option_arms()) {
        DenseBuf<Elt> C(A.rows(), B.cols(), Layout::RowMajor);
        Recorder<Elt> rec(ctx);
        const BufId bC = rec.output(view(C));
        rec.record_gemm_block(A, B, rec.whole(bC));
        Program<Elt> prog = rec.finish();
        ex.run(prog, rec.bindings(), arm.second);
        g.same(cview(C), cview(Cref), tag + " block x block " + arm.first);
        if (arm.first == "default") {
            // the M1.M2 memo of 04 Sec. 4.3: every distinct dense pair once
            uint32_t gemms = 0;
            for (const auto& n : prog.nodes) if (!n.dead && n.op == Op::GEMM) ++gemms;
            for (const auto& n : prog.nodes) if (!n.dead && n.op == Op::BATCHED_GEMM) gemms += n.batch_end - n.batch_begin;
            g.check(gemms <= prog.stats.recorded, tag + " CSE never adds products");
        }
    }
}

template<class Elt>
void gather_scatter(Executor<Elt>& ex, Elt p, const std::vector<uint32_t>& rd, const std::vector<uint32_t>& cd,
                    uint64_t seed, const std::string& tag) {
    const ModCtx<Elt> ctx(p);
    auto pool = std::make_shared<BlockPool<Elt>>();
    Rng rng(seed);
    const BlockMat<Elt> A = make_input<Elt>(rng, pool, p, Shape::Generic, rd, cd, 70, 4);
    const DenseBuf<Elt> Dref = A.to_dense(Layout::RowMajor);
    trace(tag + " gather_scatter");
    for (const auto& arm : option_arms()) {
        DenseBuf<Elt> D(A.rows(), A.cols(), Layout::ColMajor), E(A.rows(), A.cols(), Layout::RowMajor);
        Recorder<Elt> rec(ctx);
        const BufId bD = rec.output(view(D));
        const BufId bE = rec.output(view(E));
        rec.record_gather_dense(A, rec.whole(bD));           // block -> flat (column-major L1-style)
        for (uint32_t br = 0; br < A.nbr(); ++br)            // flat -> block windows (row-major)
            for (uint32_t bc = 0; bc < A.nbc(); ++bc) {
                const uint32_t dr = A.row_dim(br), dc = A.col_dim(bc);
                if (dr == 0 || dc == 0) continue;
                rec.scatter(rec.view(bD, A.row_start(br), dr, A.col_start(bc), dc),
                            rec.view(bE, A.row_start(br), dr, A.col_start(bc), dc));
            }
        const uint32_t cmp = rec.compare(rec.whole(bD), rec.whole(bE));
        Program<Elt> prog = rec.finish();
        const RunResult rr = ex.run(prog, rec.bindings(), arm.second);
        g.same(cview(D), cview(Dref), tag + " gather " + arm.first);
        g.same(cview(E), cview(Dref), tag + " scatter " + arm.first);
        g.check(rr.slots[cmp].written && rr.slots[cmp].equal, tag + " COMPARE slot equal " + arm.first);
    }
}

template<class Elt>
void op_set(Executor<Elt>& ex, Elt p, uint64_t seed, const std::string& tag) {
    const ModCtx<Elt> ctx(p);
    Rng rng(seed);
    const uint32_t m = 37, n = 29, k = 23;
    trace(tag + " op_set");
    const DenseBuf<Elt> A = random_buf<Elt>(rng, m, k, p), B = random_buf<Elt>(rng, k, n, p, Layout::ColMajor);
    const DenseBuf<Elt> Sq = random_buf<Elt>(rng, m, m, p);
    const Elt alpha = rng.nonzero<Elt>(p), beta = rng.nonzero<Elt>(p);
    // reference
    DenseBuf<Elt> R1(m, n), R2(m, m), R3(m, n), R4(m, n), R5(m, n);
    kernels::gemm<Elt>(ctx, cview(A), cview(B), view(R1), GemmOp::Set);
    for (uint32_t i = 0; i < m; ++i) for (uint32_t j = 0; j < n; ++j) R1.at(i, j) = ctx.mul(alpha, R1.at(i, j));   // alpha * A.B
    kernels::copy<Elt>(cview(Sq), view(R2));
    for (uint32_t i = 0; i < m; ++i) R2.at(i, i) = ctx.sub(R2.at(i, i), beta);                                       // Sq - beta I
    kernels::scal<Elt>(ctx, alpha, view(R2));                                                                         // alpha (Sq - beta I)
    kernels::copy<Elt>(cview(R1), view(R3)); kernels::scal<Elt>(ctx, ctx.inv(alpha), view(R3));                       // TRSM Scalar: alpha^-1 R1
    kernels::gemm<Elt>(ctx, cview(Sq), cview(R1), view(R4), GemmOp::Set);                                             // TRSM General with "inverse" Sq
    kernels::copy<Elt>(cview(R1), view(R5));                                                                          // TRSM Identity
    for (const auto& arm : option_arms()) {
        DenseBuf<Elt> O1(m, n), O2(m, m, Layout::ColMajor), O3(m, n), O4(m, n, Layout::ColMajor), O5(m, n);
        Recorder<Elt> rec(ctx);
        const BufId bA = rec.input(cview(A)), bB = rec.input(cview(B)), bSq = rec.input(cview(Sq));
        const BufId b1 = rec.output(view(O1)), b2 = rec.output(view(O2)), b3 = rec.output(view(O3)),
                    b4 = rec.output(view(O4)), b5 = rec.output(view(O5));
        rec.gemm(rec.whole(bA), rec.whole(bB), rec.whole(b1), GemmOp::Set, alpha);
        rec.scatter(rec.whole(bSq), rec.whole(b2));
        rec.axpy_identity(beta, rec.whole(b2), true);
        rec.scal(alpha, rec.whole(b2));
        rec.trsm(PivotKind::Scalar, ViewRef{}, rec.whole(b1), rec.whole(b3), ctx.inv(alpha));
        rec.trsm(PivotKind::General, rec.whole(bSq), rec.whole(b1), rec.whole(b4));
        rec.trsm(PivotKind::Identity, ViewRef{}, rec.whole(b1), rec.whole(b5));
        const uint32_t c_eq = rec.compare(rec.whole(b1), rec.whole(b5));
        const uint32_t c_ne = rec.compare(rec.whole(b1), rec.whole(b3));
        Program<Elt> prog = rec.finish();
        const RunResult rr = ex.run(prog, rec.bindings(), arm.second);
        g.same(cview(O1), cview(R1), tag + " GEMM alpha " + arm.first);
        g.same(cview(O2), cview(R2), tag + " AXPY identity + SCAL " + arm.first);
        g.same(cview(O3), cview(R3), tag + " TRSM Scalar " + arm.first);
        g.same(cview(O4), cview(R4), tag + " TRSM General " + arm.first);
        g.same(cview(O5), cview(R5), tag + " TRSM Identity " + arm.first);
        g.check(rr.slots[c_eq].equal, tag + " COMPARE equal " + arm.first);
        g.check(!rr.slots[c_ne].equal || alpha == Elt(1), tag + " COMPARE differs " + arm.first);
    }
    // CANONICALIZE: a Lazy2p buffer
    if (ctx.regime == Regime::U32_Delayed || ctx.regime == Regime::U32_Shoup) {
        DenseBuf<Elt> L(m, n, Layout::RowMajor, 0, Invariant::Lazy2p), Lref(m, n);
        for (uint32_t i = 0; i < m; ++i) for (uint32_t j = 0; j < n; ++j) {
            const Elt v = rng.elt<Elt>(p);
            L.at(i, j) = static_cast<Elt>(v + (rng.below(2) ? p : 0));
            Lref.at(i, j) = v;
        }
        Recorder<Elt> rec(ctx);
        const BufId bL = rec.output(view(L));
        rec.canonicalize(rec.whole(bL));
        Program<Elt> prog = rec.finish();
        ex.run(prog, rec.bindings(), PassOptions());
        g.same(cview(L), cview(Lref), tag + " CANONICALIZE");
    }
    // k-blocking at a tiny k_max (p = 2^31-1: k_max = 4) with k = 64, and a k > 1023 GEMM
    {
        const uint32_t kk = 64;
        const DenseBuf<Elt> Ak = random_buf<Elt>(rng, 16, kk, p), Bk = random_buf<Elt>(rng, kk, 24, p);
        DenseBuf<Elt> Rk(16, 24);
        kernels::gemm<Elt>(ctx, cview(Ak), cview(Bk), view(Rk), GemmOp::Set);
        for (bool split : {false, true}) {
            DenseBuf<Elt> Ok(16, 24);
            Recorder<Elt> rec(ctx);
            const BufId a = rec.input(cview(Ak)), b = rec.input(cview(Bk)), o = rec.output(view(Ok));
            rec.gemm(rec.whole(a), rec.whole(b), rec.whole(o), GemmOp::Set);
            Program<Elt> prog = rec.finish();
            PassOptions po; po.ksplit = split;
            ex.run(prog, rec.bindings(), po);
            g.same(cview(Ok), cview(Rk), tag + " k-blocking split=" + std::to_string(split));
            if (split) g.check(prog.stats.ksplit_nodes == (kk > ctx.k_max ? 1u : 0u), tag + " ksplit node count");
            for (const auto& nd : prog.nodes) if (!nd.dead && nd.op == Op::GEMM) g.check(nd.k_bound <= ctx.k_max, tag + " k_bound <= k_max");
        }
    }
}

template<class Elt>
void program_cache(Executor<Elt>& ex, Elt p, uint64_t seed, const std::string& tag) {
    const ModCtx<Elt> ctx(p);
    auto pool = std::make_shared<BlockPool<Elt>>();
    Rng rng(seed);
    const std::vector<uint32_t> rd(6, 17), cd(7, 17);
    const BlockMat<Elt> A = make_input<Elt>(rng, pool, p, Shape::CqLike, rd, cd, 60, 8);
    const uint32_t w = 11;
    ProgramCache<Elt> cache;
    trace(tag + " program_cache");
    ShapeKey key;
    key.row_dims = rd; key.col_dims = cd; key.pattern_hash = pattern_hash_of(A); key.d_V = 17;
    key.regime = static_cast<uint8_t>(ctx.regime); key.elt_width = sizeof(Elt); key.tag = "theta";
    g.check(cache.find(key) == nullptr, tag + " cache: initial miss");
    uint64_t recorded_hash = 0;
    {
        const DenseBuf<Elt> X = random_buf<Elt>(rng, A.cols(), w, p, Layout::ColMajor);
        DenseBuf<Elt> Y(A.rows(), w, Layout::ColMajor), Yref(A.rows(), w, Layout::ColMajor);
        Recorder<Elt> rec(ctx);
        const BufId bX = rec.input(cview(X)), bY = rec.output(view(Y));
        rec.record_gemm_dense(A, rec.whole(bX), rec.whole(bY), GemmOp::Set);
        Program<Elt> prog = rec.finish();
        recorded_hash = prog.structural_hash;
        ex.run(prog, rec.bindings(), PassOptions());
        blk::gemm_dense<Elt>(A, cview(X), view(Yref), GemmOp::Set);
        g.same(cview(Y), cview(Yref), tag + " cache: first run");
        cache.insert(key, std::move(prog));
    }
    // replay on new data of the same shape: equals a fresh recording
    for (int rep = 0; rep < 3; ++rep) {
        const BlockMat<Elt> A2 = resample_data<Elt>(rng, A, pool);
        g.check(pattern_hash_of(A2) == key.pattern_hash, tag + " cache: resampled pattern hash equal");
        const DenseBuf<Elt> X2 = random_buf<Elt>(rng, A2.cols(), w, p, Layout::ColMajor);
        DenseBuf<Elt> Yc(A2.rows(), w, Layout::ColMajor), Yf(A2.rows(), w, Layout::ColMajor), Yref(A2.rows(), w, Layout::ColMajor);
        Recorder<Elt> fresh(ctx);
        const BufId bX = fresh.input(cview(X2)), bY = fresh.output(view(Yf));
        fresh.record_gemm_dense(A2, fresh.whole(bX), fresh.whole(bY), GemmOp::Set);
        Program<Elt> fp = fresh.finish();
        g.check(fp.structural_hash == recorded_hash, tag + " cache: fresh recording has the recorded structure");
        Bindings<Elt> b = fresh.bindings();
        b.outputs[0] = view(Yc);
        std::shared_ptr<const Program<Elt>> cached = cache.find(key);
        g.check(cached != nullptr, tag + " cache: hit");
        if (cached) ex.run(*cached, b);
        ex.run(fp, fresh.bindings(), PassOptions());
        blk::gemm_dense<Elt>(A2, cview(X2), view(Yref), GemmOp::Set);
        g.same(cview(Yc), cview(Yref), tag + " cache: replay == direct");
        g.same(cview(Yf), cview(Yref), tag + " cache: fresh == direct");
    }
    // a different pattern hash must miss (the D4 cache-poisoning caveat)
    {
        ShapeKey k2 = key; k2.pattern_hash ^= 0x1234567ull;
        g.check(cache.find(k2) == nullptr, tag + " cache: changed pattern hash misses");
        ShapeKey k3 = key; k3.extra = 1;
        g.check(cache.find(k3) == nullptr, tag + " cache: changed extra misses");
        // a changed pattern (one block dropped) hashes differently
        BlockBuilder<Elt> bb(pool, rd, cd, p);
        bool dropped = false;
        for (uint32_t br = 0; br < A.nbr(); ++br)
            for (uint32_t k = A.row_begin(br); k < A.row_end(br); ++k) {
                if (!dropped && A.ref_at(k).has_data()) { dropped = true; continue; }
                bb.set(br, A.col_at(k), A.ref_at(k));
            }
        const BlockMat<Elt> A3 = bb.finalize();
        g.check(pattern_hash_of(A3) != key.pattern_hash, tag + " cache: dropped block changes the pattern hash");
        // wrong-shape bindings are rejected
        bool threw = false;
        try {
            Bindings<Elt> bad;
            const DenseBuf<Elt> Xs = random_buf<Elt>(rng, A.cols(), w + 1, p, Layout::ColMajor);
            DenseBuf<Elt> Ys(A.rows(), w + 1, Layout::ColMajor);
            std::shared_ptr<const Program<Elt>> cached = cache.find(key);
            bad.inputs.assign(cached->n_inputs, cview(Xs)); bad.outputs.assign(1, view(Ys));
            ex.run(*cached, bad);
        } catch (const std::logic_error&) { threw = true; }
        g.check(threw, tag + " cache: mismatched binding shape is refused");
    }
    const auto st = cache.stats();
    g.check(st.hits == 4 && st.inserts == 1, tag + " cache: stats hits=4 inserts=1");
}

template<class Elt>
void run_width(Executor<Elt>& ex, const std::vector<Elt>& primes, const std::string& w, bool big) {
    uint64_t seed = 0x7A9E;
    for (Elt p : primes) {
        const std::string tag = "[" + w + " p=" + std::to_string(p) + "]";
        theta_like<Elt>(ex, p, {5, 5, 5}, {5, 5, 5, 5}, Shape::CqLike, 60, 7, seed++, tag + " 3x4 d=5");
        theta_like<Elt>(ex, p, {3, 7, 2}, {5, 1, 6}, Shape::Generic, 70, 4, seed++, tag + " non-uniform");
        theta_like<Elt>(ex, p, {3, 0, 2}, {5, 0, 6}, Shape::Generic, 70, 3, seed++, tag + " zero-width");
        theta_like<Elt>(ex, p, std::vector<uint32_t>(6, 17), std::vector<uint32_t>(7, 17), Shape::ZqLike, 40, 13, seed++, tag + " 6x7 d=17 Zq");
        theta_like<Elt>(ex, p, std::vector<uint32_t>(6, 17), std::vector<uint32_t>(7, 17), Shape::Generic, 40, 13, seed++, tag + " 6x7 d=17 generic");
        block_product<Elt>(ex, p, {4, 4, 4}, {4, 4, 4, 4}, {4, 4}, seed++, tag + " 3x4x2 d=4");
        block_product<Elt>(ex, p, {3, 7, 2}, {5, 1, 6}, {2, 4, 3}, seed++, tag + " non-uniform");
        gather_scatter<Elt>(ex, p, {3, 7, 2}, {5, 1, 6}, seed++, tag + " non-uniform");
        gather_scatter<Elt>(ex, p, std::vector<uint32_t>(6, 17), std::vector<uint32_t>(7, 17), seed++, tag + " 6x7 d=17");
        op_set<Elt>(ex, p, seed++, tag);
        program_cache<Elt>(ex, p, seed++, tag);
        if (big) {
            theta_like<Elt>(ex, p, std::vector<uint32_t>(11, 129), std::vector<uint32_t>(11, 129), Shape::CqLike, 30, 33, seed++, tag + " 11x11 d=129 Cq");
            theta_like<Elt>(ex, p, std::vector<uint32_t>(6, 65), std::vector<uint32_t>(7, 65), Shape::Generic, 40, 40, seed++, tag + " 6x7 d=65 generic");
            block_product<Elt>(ex, p, std::vector<uint32_t>(6, 65), std::vector<uint32_t>(7, 65), std::vector<uint32_t>(5, 65), seed++, tag + " 6x7x5 d=65");
        }
    }
}

} // namespace

uint32_t EngineTapeValidation() {
    const auto t0 = std::chrono::steady_clock::now();
    g = Checker();
    std::cout << "=== EngineTapeValidation (record -> passes -> execute vs direct blk:: / kernels:: execution) ===" << std::endl;
    try {
        Executor<uint32_t> ex32;
        Executor<uint64_t> ex64;
        run_width<uint32_t>(ex32, {134217757u}, "u32", true);
        run_width<uint32_t>(ex32, {2u, 3u, 7u, 2147483647u, 4294967291u}, "u32", false);
        run_width<uint64_t>(ex64, {134217757ull, 2305843009213693951ull}, "u64", false);
    } catch (const std::exception& e) {
        std::cout << "  EXCEPTION: " << e.what() << std::endl;
        ++g.fails;
    }
    std::cout << "EngineTapeValidation: " << g.checks << " checks, " << g.fails << " failed, "
              << seconds_since(t0) << " s (pool " << ThreadPool::global().size() << " threads)" << std::endl;
    return g.fails == 0 ? 0 : 1;
}
