// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// EngineCrossDeviceValidation.cpp -- the tape under ENGINE_DEVICE in
// {cpu, gpu, hybrid} (matrix-engine Phase 5 Stage B, plan Sec. 4.3).
//
// The core's pipeline-shaped programs (EngineDeterminismCoreValidation's
// family: Theta-like gemm_dense with shared pool operands, block x block,
// the elimination driver rref / kernel / factor_basis / solve, gather /
// scatter with COMPARE) plus the op set (scaled GEMM, TRSM kinds, AXPY with
// the identity source, SCAL, CANONICALIZE), a scaled batched program, and
// k > k_max GEMMs in every regime, each run under every device policy x
// thread count x batching x graph-capture mode, with the ENGINE_VERIFY_DEVICE
// re-execution switched on for every device arm.  FNV-64 hashes of every
// output must be identical to the cpu arm's; the gpu arm must actually have
// executed nodes on the device (a vacuous pass is a failure).
//
// The golden grid under ENGINE_DEVICE is checked by GoldenRegressionValidation itself
// (the consumers record on the tape since Phase 4 Stage B); see the closing line below.
#include "EngineCrossDeviceValidation.h"

#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <map>
#include <string>

#include "../Cuda/CudaCtx.h"
#include "../ModArithmetic/BlockKernels.h"
#include "../ModArithmetic/Kernels.h"
#include "../ModArithmetic/RetiredDeviceLevers.h"   // (S25 LEVERS-25) gpu_lean_lever()
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
namespace cu = engine::cuda;

Checker g;

struct EnvGuard {
    std::string name, old; bool had;
    EnvGuard(const char* n, const char* v) : name(n) {
        const char* o = std::getenv(n); had = o != nullptr; if (had) old = o;
        setenv(n, v, 1);
    }
    ~EnvGuard() { if (had) setenv(name.c_str(), old.c_str(), 1); else unsetenv(name.c_str()); }
};

struct Setting { const char* device; uint32_t threads; bool batch; int graph; bool verify; std::string name; bool lean = true; };

// verify = the ENGINE_VERIFY_DEVICE re-execution.  It makes the host copies
// current before every GPU level, so the planned D2H nodes become no-ops;
// the verify=0 arms exercise the real downloads.
std::vector<Setting> settings() {
    std::vector<Setting> s;
    s.push_back({"cpu", 16, true, 1, false, "cpu threads=16"});
    s.push_back({"cpu", 1, false, 1, false, "cpu threads=1 batch=0"});
    // (slot-23 P7) the first three gpu arms run with ENGINE_GPU_LEAN=0 (since S25 LEVERS-25 the Scoped override, not the env), so the memory-only programs of the
    // family keep their device coverage (AXPY / SCAL / CANONICALIZE / GATHER / SCATTER kernels); the two
    // threads=1 arms run under the default VRAM-lean rule (those programs on the host).
    s.push_back({"gpu", 16, true, 0, true, "gpu graph=0 verify lean=0", false});
    s.push_back({"gpu", 16, true, 2, true, "gpu graph=2 verify lean=0", false});
    s.push_back({"gpu", 16, true, 1, false, "gpu graph=1 (no verify) lean=0", false});
    s.push_back({"gpu", 1, false, 2, false, "gpu threads=1 batch=0 graph=2 (no verify)"});
    s.push_back({"gpu", 1, false, 1, true, "gpu threads=1 batch=0 graph=1 verify"});
    s.push_back({"hybrid", 16, true, 1, true, "hybrid verify"});
    s.push_back({"hybrid", 16, true, 1, false, "hybrid (no verify)"});
    s.push_back({"hybrid", 4, false, 2, true, "hybrid threads=4 batch=0 graph=2 verify"});
    return s;
}

PassOptions opts_for(const Setting& st) {
    PassOptions o;
    o.batch = st.batch;
    o.policy = (std::string(st.device) == "gpu") ? DevicePolicy::Gpu
             : (std::string(st.device) == "hybrid") ? DevicePolicy::Hybrid : DevicePolicy::Cpu;
    return o;
}

template<class Elt>
uint64_t hash_result(const RunResult& rr) {
    uint64_t h = fnv_init();
    for (const ResultSlot& s : rr.slots) {
        h = fnv_mix(h, s.written ? 1 : 0); h = fnv_mix(h, s.rank); h = fnv_mix(h, s.equal ? 1 : 0);
        for (uint32_t p : s.pivots) h = fnv_mix(h, p);
    }
    return h;
}

// The program family: one hash per program (outputs and result slots).
template<class Elt>
std::vector<uint64_t> run_family(Elt p, uint64_t seed, const Setting& st, ThreadPool& pool, std::vector<std::string>* names) {
    const ModCtx<Elt> ctx(p);
    Executor<Elt> ex(&pool);
    ex.set_verify_device(st.verify);
    const PassOptions opts = opts_for(st);
    std::vector<uint64_t> hashes;
    auto push = [&](uint64_t h, const char* what) { hashes.push_back(h); if (names) names->push_back(what); };
    auto bpool = std::make_shared<BlockPool<Elt>>();
    Rng rng(seed);
    // 1. Theta-like gemm_dense: 11x11 d=129 Cq (8 shared operands), col-major multi-vectors; run twice (replay -> graph)
    {
        const BlockMat<Elt> A = make_input<Elt>(rng, bpool, p, Shape::CqLike, std::vector<uint32_t>(11, 129), std::vector<uint32_t>(11, 129), 30, 8);
        const DenseBuf<Elt> X = random_buf<Elt>(rng, A.cols(), 33, p, Layout::ColMajor);
        DenseBuf<Elt> Y(A.rows(), 33, Layout::ColMajor);
        Recorder<Elt> rec(ctx);
        const BufId bX = rec.input(cview(X)), bY = rec.output(view(Y));
        rec.record_gemm_dense(A, rec.whole(bX), rec.whole(bY), GemmOp::Set);
        Program<Elt> prog = rec.finish();
        ex.run(prog, rec.bindings(), opts);
        push(hash_view(cview(Y)), "theta-like gemm_dense");
        DenseBuf<Elt> Y2(A.rows(), 33, Layout::ColMajor);
        Bindings<Elt> b = rec.bindings(); b.outputs[0] = view(Y2);
        ex.run(static_cast<const Program<Elt>&>(prog), b);
        push(hash_view(cview(Y2)), "theta-like gemm_dense (replay)");
        g.check(hash_view(cview(Y2)) == hash_view(cview(Y)), std::string("replay == first run (") + st.name + ")");
    }
    // 2. block x block, generic kinds (transposed windows, scalars, affine)
    {
        const BlockMat<Elt> A = make_input<Elt>(rng, bpool, p, Shape::Generic, std::vector<uint32_t>(6, 65), std::vector<uint32_t>(7, 65), 45, 6);
        const BlockMat<Elt> B = make_input<Elt>(rng, bpool, p, Shape::Generic, std::vector<uint32_t>(7, 65), std::vector<uint32_t>(5, 65), 45, 6);
        DenseBuf<Elt> C(A.rows(), B.cols());
        Recorder<Elt> rec(ctx);
        const BufId bC = rec.output(view(C));
        rec.record_gemm_block(A, B, rec.whole(bC));
        Program<Elt> prog = rec.finish();
        ex.run(prog, rec.bindings(), opts);
        push(hash_view(cview(C)), "block x block");
    }
    // 3. elimination driver (reads ENGINE_DEVICE / ENGINE_BATCH from the environment)
    for (int which = 0; which < 2; ++which) {
        const BlockMat<Elt> A = which == 0
            ? make_input<Elt>(rng, bpool, p, Shape::CqLike, std::vector<uint32_t>(13, 129), std::vector<uint32_t>(7, 129), 18, 8)
            : make_input<Elt>(rng, bpool, p, Shape::Dependent, std::vector<uint32_t>(6, 65), std::vector<uint32_t>(7, 65), 25, 6);
        const blk::BlockRrefResult<Elt> res = rref<Elt>(ex, A);
        uint64_t h = fnv_mix(fnv_init(), res.rank);
        for (uint32_t c : res.pivots) h = fnv_mix(h, c);
        h = fnv_mix(h, hash_view(cview(res.R)));
        push(h, which == 0 ? "rref 13x7 d=129 Cq" : "rref 6x7 d=65 dependent");
        const BlockMat<Elt> K = blk::kernel_basis<Elt>(A, res, Layout::ColMajor);
        push(hash_view(cview(K.to_dense(Layout::ColMajor))), "kernel_basis");
        const std::vector<uint32_t> row_dims = A.col_dims();
        const uint32_t a = A.cols();
        DenseBuf<Elt> Rl = random_buf<Elt>(rng, a, 3, p, Layout::ColMajor), G = random_buf<Elt>(rng, a, 9, p, Layout::ColMajor);
        for (uint32_t i = 0; i < a; ++i) G.at(i, 2) = Rl.at(i, 1);
        const blk::FactorBasisResult<Elt> fb = factor_basis<Elt>(ex, bpool, p, row_dims, cview(Rl), cview(G));
        uint64_t hf = fnv_mix(fnv_init(), fb.independent_relations);
        for (uint32_t c : fb.pivots) hf = fnv_mix(hf, c);
        hf = fnv_mix(hf, hash_view(cview(fb.projector)));
        hf = fnv_mix(hf, hash_view(cview(fb.basis.to_dense(Layout::ColMajor))));
        push(hf, "factor_basis");
        DenseBuf<Elt> coef = random_buf<Elt>(rng, 9, 4, p), Y(a, 4, Layout::ColMajor);
        kernels::gemm<Elt>(ctx, cview(G), cview(coef), view(Y), GemmOp::Set);
        const DenseBuf<Elt> sol = solve_multi_rhs<Elt>(ex, bpool, p, row_dims, cview(G), cview(Y));
        push(hash_view(cview(sol)), "solve_multi_rhs");
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
        const RunResult rr = ex.run(prog, rec.bindings(), opts);
        push(hash_view(cview(E)) ^ (rr.slots[cmp].equal ? 1u : 0u), "gather/scatter/COMPARE");
        g.check(rr.slots[cmp].equal, std::string("COMPARE equal (") + st.name + ")");
    }
    // 5. the op set: scaled GEMM (Set / Add / Sub), TRSM kinds, AXPY identity, SCAL, CANONICALIZE, k > k_max
    {
        const uint32_t m = 37, n = 29, k = 23;
        const DenseBuf<Elt> A = random_buf<Elt>(rng, m, k, p), B = random_buf<Elt>(rng, k, n, p, Layout::ColMajor);
        const DenseBuf<Elt> Sq = random_buf<Elt>(rng, m, m, p);
        const Elt alpha = rng.nonzero<Elt>(p), beta = rng.nonzero<Elt>(p);
        DenseBuf<Elt> O1(m, n), O1b(m, n, Layout::ColMajor), O1c(m, n), O2(m, m, Layout::ColMajor), O3(m, n), O4(m, n, Layout::ColMajor), O5(m, n);
        Recorder<Elt> rec(ctx);
        const BufId bA = rec.input(cview(A)), bB = rec.input(cview(B)), bSq = rec.input(cview(Sq));
        const BufId b1 = rec.output(view(O1)), b1b = rec.output(view(O1b)), b1c = rec.output(view(O1c)), b2 = rec.output(view(O2)),
                    b3 = rec.output(view(O3)), b4 = rec.output(view(O4)), b5 = rec.output(view(O5));
        rec.gemm(rec.whole(bA), rec.whole(bB), rec.whole(b1), GemmOp::Set, alpha);
        rec.scatter(rec.whole(bSq).sub(0, m, 0, n), rec.whole(b1b));
        rec.gemm(rec.whole(bA), rec.whole(bB), rec.whole(b1b), GemmOp::Add, alpha);
        rec.scatter(rec.whole(bSq).sub(0, m, 0, n), rec.whole(b1c));
        rec.gemm(rec.whole(bA), rec.whole(bB), rec.whole(b1c), GemmOp::Sub, beta);
        rec.scatter(rec.whole(bSq), rec.whole(b2));
        rec.axpy_identity(beta, rec.whole(b2), true);
        rec.scal(alpha, rec.whole(b2));
        rec.trsm(PivotKind::Scalar, ViewRef{}, rec.whole(b1), rec.whole(b3), ctx.inv(alpha));
        rec.trsm(PivotKind::General, rec.whole(bSq), rec.whole(b1), rec.whole(b4));
        rec.trsm(PivotKind::Identity, ViewRef{}, rec.whole(b1), rec.whole(b5));
        const uint32_t c_eq = rec.compare(rec.whole(b1), rec.whole(b5));
        Program<Elt> prog = rec.finish();
        const RunResult rr = ex.run(prog, rec.bindings(), opts);
        uint64_t h = hash_view(cview(O1)) ^ (hash_view(cview(O1b)) * 3) ^ (hash_view(cview(O1c)) * 5) ^ (hash_view(cview(O2)) * 7) ^
                     (hash_view(cview(O3)) * 11) ^ (hash_view(cview(O4)) * 13) ^ (hash_view(cview(O5)) * 17) ^ hash_result<Elt>(rr);
        push(h, "op set");
        g.check(rr.slots[c_eq].equal, std::string("op set COMPARE (") + st.name + ")");
    }
    // 6. scaled batched program (alpha != 1 on every instance) accumulated into a row-major output
    {
        const uint32_t d = 33, nb = 12, terms = 60;
        std::vector<DenseBuf<Elt>> rho;
        for (uint32_t i = 0; i < 5; ++i) rho.push_back(random_buf<Elt>(rng, d, d, p));
        const DenseBuf<Elt> X = random_buf<Elt>(rng, nb * d, 7, p, Layout::ColMajor);
        DenseBuf<Elt> Y(nb * d, 7);
        Recorder<Elt> rec(ctx);
        std::vector<BufId> bR;
        for (uint32_t i = 0; i < 5; ++i) bR.push_back(rec.input(cview(rho[i]), 200 + i));
        const BufId bX = rec.input(cview(X)), bY = rec.output(view(Y));
        rec.zero(rec.whole(bY));
        const Elt c1 = rng.nonzero<Elt>(p), c2 = rng.nonzero<Elt>(p);
        for (uint32_t t = 0; t < terms; ++t) {
            const uint32_t a = (t * 7 + 3) % 5, b = (t * 5 + 1) % nb;
            rec.gemm_ref(Elt(0), (t % 2) ? c1 : c2, rec.whole(bR[a]), Elt(0), Elt(1), rec.view(bX, b * d, d, 0, 7),
                         rec.view(bY, b * d, d, 0, 7), d, d, 7, (t % 3) ? GemmOp::Add : GemmOp::Sub);
        }
        Program<Elt> prog = rec.finish();
        ex.run(prog, rec.bindings(), opts);
        push(hash_view(cview(Y)), "scaled batched");
    }
    // 7. k > k_max GEMM (k = 4200: at least 4 k-chunks at ell = 134217757, 1050 at 2^31-1, 4200 at 2^32-5)
    {
        const uint32_t kk = 4200, mm = 24, nn = 40;
        const DenseBuf<Elt> Ak = random_buf<Elt>(rng, mm, kk, p), Bk = random_buf<Elt>(rng, kk, nn, p);
        DenseBuf<Elt> Ok(mm, nn);
        Recorder<Elt> rec(ctx);
        const BufId a = rec.input(cview(Ak)), b = rec.input(cview(Bk)), o = rec.output(view(Ok));
        rec.gemm(rec.whole(a), rec.whole(b), rec.whole(o), GemmOp::Set);
        Program<Elt> prog = rec.finish();
        ex.run(prog, rec.bindings(), opts);
        push(hash_view(cview(Ok)), "k=4200 GEMM");
        DenseBuf<Elt> Rk(mm, nn);
        kernels::gemm<Elt>(ctx, cview(Ak), cview(Bk), view(Rk), GemmOp::Set);
        g.check(same_entries(cview(Ok), cview(Rk)), std::string("k=4200 GEMM == kernels::gemm (") + st.name + ")");
    }
    const DeviceStats& ds = ex.device_stats();
    if (std::string(st.device) == "gpu") {
        g.check(ds.gpu_nodes > 0, std::string("gpu arm executed nodes on the device (") + st.name + ")");
        if (st.verify) g.check(ds.verified_nodes == ds.gpu_nodes, std::string("every GPU node was verified (") + st.name + ")");
        else g.check(ds.d2h > 0, std::string("downloads executed on the unverified arm (") + st.name + ")");
        if (st.graph == 2) g.check(ds.graph_launches > 0, std::string("graph replay used (") + st.name + ")");
    }
    if (std::getenv("ENGINE_TAPE_TRACE"))
        std::cout << "    " << st.name << ": gpu nodes " << ds.gpu_nodes << ", levels " << ds.gpu_levels << ", direct " << ds.direct_launches
                  << ", graph " << ds.graph_launches << " (" << ds.captures << " captures), h2d " << ds.h2d << " (" << ds.h2d_bytes
                  << " B), d2h " << ds.d2h << " (" << ds.d2h_bytes << " B), skipped " << ds.skipped_transfers
                  << ", verified " << ds.verified_nodes << ", fallback " << ex.stats().fallback_nodes << std::endl;
    return hashes;
}

template<class Elt>
void run_prime(Elt p, uint64_t seed, const std::string& tag) {
    std::vector<uint64_t> reference;
    std::vector<std::string> names;
    std::string ref_name;
    for (const Setting& st : settings()) {
        EnvGuard d("ENGINE_DEVICE", st.device);
        EnvGuard b("ENGINE_BATCH", st.batch ? "1" : "0");
        EnvGuard gr("ENGINE_CUDA_GRAPH", st.graph == 0 ? "0" : st.graph == 2 ? "2" : "1");
        engine::lever::EnvLever::Scoped ln(engine::lever::gpu_lean_lever(), st.lean);   // (S25 LEVERS-25) was the env arm
        ThreadPool pool(st.threads);
        const auto t0 = std::chrono::steady_clock::now();
        std::vector<std::string> nm;
        std::vector<uint64_t> h;
        try {
            h = run_family<Elt>(p, seed, st, pool, reference.empty() ? &names : &nm);
        } catch (const std::exception& e) {
            g.check(false, tag + " '" + st.name + "': exception " + e.what());
            continue;
        }
        const double secs = seconds_since(t0);
        if (reference.empty()) { reference = h; ref_name = st.name; }
        const bool same = (h == reference);
        g.check(same, tag + " hashes under '" + st.name + "' == '" + ref_name + "'");
        if (!same && h.size() == reference.size())
            for (size_t i = 0; i < h.size(); ++i)
                if (h[i] != reference[i]) std::cout << "    differs: " << names[i] << std::endl;
        std::cout << "  " << tag << " " << std::left << std::setw(42) << st.name << ": " << h.size() << " hashes, "
                  << (same ? "identical" : "DIFFERENT") << ", " << std::fixed << std::setprecision(3) << secs << " s" << std::endl;
    }
}

} // namespace

uint32_t EngineCrossDeviceValidation() {
    const auto t0 = std::chrono::steady_clock::now();
    g = Checker();
    std::cout << "=== EngineCrossDeviceValidation (ENGINE_DEVICE cpu / gpu / hybrid x threads x batch x graph: hashes identical, GPU nodes verified) ===" << std::endl;
    if (!cu::available()) {
        std::cout << "[SKIP] EngineCrossDeviceValidation: no usable CUDA device (" << cu::unavailable_reason()
                  << ") -- the device arm cannot be exercised on this machine" << std::endl;
        return 0;
    }
    try {
        const cu::DeviceInfo di = cu::device_info();
        std::cout << "  device: " << di.name << " (cc " << di.cc_major << "." << di.cc_minor << ", " << di.sms << " SMs)" << std::endl;
        run_prime<uint32_t>(134217757u, 0xC0DE, "[u32 p=134217757]");
        run_prime<uint32_t>(2147483647u, 0xC0DF, "[u32 p=2^31-1]");
        run_prime<uint32_t>(4294967291u, 0xC0E0, "[u32 p=2^32-5]");
        run_prime<uint32_t>(7u, 0xC0E1, "[u32 p=7]");
        run_prime<uint64_t>(2305843009213693951ull, 0xC0E2, "[u64 p=2^61-1]");
        run_prime<uint64_t>(134217757ull, 0xC0E3, "[u64 p=134217757]");
    } catch (const std::exception& e) {
        std::cout << "  EXCEPTION: " << e.what() << std::endl;
        ++g.fails;
    }
    std::cout << "  golden grid under ENGINE_DEVICE: covered -- the consumers record on the tape since Phase 4 "
                 "Stage B, and `GOLDEN_STRICT=1 ./hacs GoldenRegressionValidation` is green under "
                 "ENGINE_DEVICE = cpu / gpu / hybrid (tier A 9446 failing 0, tier B 4603 mismatching 0 in each)"
              << std::endl;
    std::cout << "EngineCrossDeviceValidation: " << g.checks << " checks, " << g.fails << " failed, " << seconds_since(t0) << " s" << std::endl;
    return g.fails == 0 ? 0 : 1;
}
