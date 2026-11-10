// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// EngineDeterminismCoreValidation.cpp -- the same programs under
// ENGINE_THREADS in {1, 4, 16, 32} x ENGINE_BATCH in {0, 1} (plus the arena /
// layout / device-model switches), FNV-64 hashes of every output identical
// (plan D4 determinism guarantee, axis "schedule / thread count" of D7).
#include "EngineDeterminismCoreValidation.h"

#include <iostream>
#include <map>

#include "../ModArithmetic/BlockKernels.h"
#include "EngineTestUtil.h"
#include "Executor.h"
#include "Passes.h"
#include "Recorder.h"
#include "TapeElim.h"
#include "ThreadPool.h"

namespace {

using namespace engine;
using namespace engine::tape;
using namespace engine::tape::test;
using kernels::GemmOp;

Checker g;

struct Setting { uint32_t threads; PassOptions opts; std::string name; };

std::vector<Setting> settings() {
    std::vector<Setting> s;
    for (uint32_t t : {1u, 4u, 16u, 32u})
        for (bool batch : {true, false}) {
            PassOptions o; o.batch = batch;
            s.push_back({t, o, "threads=" + std::to_string(t) + " batch=" + std::to_string(batch)});
        }
    { PassOptions o; o.arena = false; s.push_back({32, o, "threads=32 arena=0"}); }
    { PassOptions o; o.layout = false; s.push_back({16, o, "threads=16 layout=0"}); }
    { PassOptions o; o.device_model = true; o.device_policy = Device::GPU; s.push_back({32, o, "threads=32 device=hybrid"}); }
    { PassOptions o; o.cse = false; o.fuse = false; s.push_back({32, o, "threads=32 cse=0 fuse=0"}); }
    return s;
}

// Every program family produces one hash per setting; all must agree.
template<class Elt>
std::vector<uint64_t> run_family(Elt p, uint64_t seed, const Setting& st, ThreadPool& pool) {
    const ModCtx<Elt> ctx(p);
    Executor<Elt> ex(&pool);
    std::vector<uint64_t> hashes;
    auto bpool = std::make_shared<BlockPool<Elt>>();
    Rng rng(seed);
    // 1. Theta-like gemm_dense on an 11x11 d=129 Cq matrix (8 shared operands), col-major multi-vectors
    {
        const BlockMat<Elt> A = make_input<Elt>(rng, bpool, p, Shape::CqLike, std::vector<uint32_t>(11, 129), std::vector<uint32_t>(11, 129), 30, 8);
        const DenseBuf<Elt> X = random_buf<Elt>(rng, A.cols(), 33, p, Layout::ColMajor);
        DenseBuf<Elt> Y(A.rows(), 33, Layout::ColMajor);
        Recorder<Elt> rec(ctx);
        const BufId bX = rec.input(cview(X)), bY = rec.output(view(Y));
        rec.record_gemm_dense(A, rec.whole(bX), rec.whole(bY), GemmOp::Set);
        Program<Elt> prog = rec.finish();
        ex.run(prog, rec.bindings(), st.opts);
        hashes.push_back(hash_view(cview(Y)));
    }
    // 2. block x block, generic kinds
    {
        const BlockMat<Elt> A = make_input<Elt>(rng, bpool, p, Shape::Generic, std::vector<uint32_t>(6, 65), std::vector<uint32_t>(7, 65), 45, 6);
        const BlockMat<Elt> B = make_input<Elt>(rng, bpool, p, Shape::Generic, std::vector<uint32_t>(7, 65), std::vector<uint32_t>(5, 65), 45, 6);
        DenseBuf<Elt> C(A.rows(), B.cols());
        Recorder<Elt> rec(ctx);
        const BufId bC = rec.output(view(C));
        rec.record_gemm_block(A, B, rec.whole(bC));
        Program<Elt> prog = rec.finish();
        ex.run(prog, rec.bindings(), st.opts);
        hashes.push_back(hash_view(cview(C)));
    }
    // 3. elimination: rref / kernel / factor_basis / solve on a 13x7 d=129 Cq matrix and a dependent 6x7 d=65
    for (int which = 0; which < 2; ++which) {
        const BlockMat<Elt> A = which == 0
            ? make_input<Elt>(rng, bpool, p, Shape::CqLike, std::vector<uint32_t>(13, 129), std::vector<uint32_t>(7, 129), 18, 8)
            : make_input<Elt>(rng, bpool, p, Shape::Dependent, std::vector<uint32_t>(6, 65), std::vector<uint32_t>(7, 65), 25, 6);
        // the executor reads the pass options from the environment inside the driver: set them
        // through the process-wide switches for this setting
        const blk::BlockRrefResult<Elt> res = rref<Elt>(ex, A);
        uint64_t h = fnv_mix(fnv_init(), res.rank);
        for (uint32_t c : res.pivots) h = fnv_mix(h, c);
        h = fnv_mix(h, hash_view(cview(res.R)));
        hashes.push_back(h);
        const BlockMat<Elt> K = blk::kernel_basis<Elt>(A, res, Layout::ColMajor);
        hashes.push_back(hash_view(cview(K.to_dense(Layout::ColMajor))));
        const std::vector<uint32_t> row_dims = A.col_dims();
        const uint32_t a = A.cols();
        DenseBuf<Elt> Rl = random_buf<Elt>(rng, a, 3, p, Layout::ColMajor), G = random_buf<Elt>(rng, a, 9, p, Layout::ColMajor);
        for (uint32_t i = 0; i < a; ++i) G.at(i, 2) = Rl.at(i, 1);
        const blk::FactorBasisResult<Elt> fb = factor_basis<Elt>(ex, bpool, p, row_dims, cview(Rl), cview(G));
        uint64_t hf = fnv_mix(fnv_init(), fb.independent_relations);
        for (uint32_t c : fb.pivots) hf = fnv_mix(hf, c);
        hf = fnv_mix(hf, hash_view(cview(fb.projector)));
        hf = fnv_mix(hf, hash_view(cview(fb.basis.to_dense(Layout::ColMajor))));
        hashes.push_back(hf);
        DenseBuf<Elt> coef = random_buf<Elt>(rng, 9, 4, p), Y(a, 4, Layout::ColMajor);
        kernels::gemm<Elt>(ctx, cview(G), cview(coef), view(Y), GemmOp::Set);
        const DenseBuf<Elt> sol = solve_multi_rhs<Elt>(ex, bpool, p, row_dims, cview(G), cview(Y));
        hashes.push_back(hash_view(cview(sol)));
    }
    // 4. gather / scatter round trip with COMPARE
    {
        const BlockMat<Elt> A = make_input<Elt>(rng, bpool, p, Shape::Generic, std::vector<uint32_t>(6, 65), std::vector<uint32_t>(7, 65), 60, 4);
        DenseBuf<Elt> D(A.rows(), A.cols(), Layout::ColMajor), E(A.rows(), A.cols());
        Recorder<Elt> rec(ctx);
        const BufId bD = rec.output(view(D)), bE = rec.output(view(E));
        rec.record_gather_dense(A, rec.whole(bD));
        rec.scatter(rec.whole(bD), rec.whole(bE));
        const uint32_t cmp = rec.compare(rec.whole(bD), rec.whole(bE));
        Program<Elt> prog = rec.finish();
        const RunResult rr = ex.run(prog, rec.bindings(), st.opts);
        hashes.push_back(hash_view(cview(E)) ^ (rr.slots[cmp].equal ? 1u : 0u));
    }
    return hashes;
}

// The elimination driver takes its options from the environment; the
// validation drives ENGINE_BATCH / ENGINE_ARENA through setenv for its arms.
struct EnvGuard {
    std::string name, old; bool had;
    EnvGuard(const char* n, const char* v) : name(n) {
        const char* o = std::getenv(n); had = o != nullptr; if (had) old = o;
        setenv(n, v, 1);
    }
    ~EnvGuard() { if (had) setenv(name.c_str(), old.c_str(), 1); else unsetenv(name.c_str()); }
};

template<class Elt>
void run_prime(Elt p, uint64_t seed, const std::string& tag) {
    std::vector<uint64_t> reference;
    std::string ref_name;
    for (const Setting& st : settings()) {
        EnvGuard b("ENGINE_BATCH", st.opts.batch ? "1" : "0");
        EnvGuard a("ENGINE_ARENA", st.opts.arena ? "1" : "0");
        EnvGuard l("ENGINE_LAYOUT", st.opts.layout ? "1" : "0");
        EnvGuard c("ENGINE_CSE", st.opts.cse ? "1" : "0");
        EnvGuard f("ENGINE_FUSE", st.opts.fuse ? "1" : "0");
        EnvGuard d("ENGINE_DEVICE", st.opts.device_model ? "hybrid" : "cpu");
        ThreadPool pool(st.threads);
        const auto t0 = std::chrono::steady_clock::now();
        const std::vector<uint64_t> h = run_family<Elt>(p, seed, st, pool);
        const double secs = seconds_since(t0);
        if (reference.empty()) { reference = h; ref_name = st.name; }
        bool same = (h == reference);
        g.check(same, tag + " hashes under '" + st.name + "' == '" + ref_name + "'");
        std::cout << "  " << tag << " " << st.name << ": " << h.size() << " hashes, "
                  << (same ? "identical" : "DIFFERENT") << ", " << secs << " s" << std::endl;
    }
}

} // namespace

uint32_t EngineDeterminismCoreValidation() {
    const auto t0 = std::chrono::steady_clock::now();
    g = Checker();
    std::cout << "=== EngineDeterminismCoreValidation (ENGINE_THREADS x ENGINE_BATCH x arena/layout/device: FNV-64 hashes identical) ===" << std::endl;
    try {
        run_prime<uint32_t>(134217757u, 0xD37E, "[u32 p=134217757]");
        run_prime<uint32_t>(2147483647u, 0xD37F, "[u32 p=2^31-1]");
        run_prime<uint64_t>(2305843009213693951ull, 0xD380, "[u64 p=2^61-1]");
    } catch (const std::exception& e) {
        std::cout << "  EXCEPTION: " << e.what() << std::endl;
        ++g.fails;
    }
    std::cout << "EngineDeterminismCoreValidation: " << g.checks << " checks, " << g.fails << " failed, " << seconds_since(t0) << " s" << std::endl;
    return g.fails == 0 ? 0 : 1;
}
