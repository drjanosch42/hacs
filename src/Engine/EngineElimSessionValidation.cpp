// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// EngineElimSessionValidation.cpp -- slot-18 WP-19a (GPU G1): the elimination
// session (DeviceBackend.h / TapeElim.cpp, ENGINE_API.md Sec. 12.8) and the
// device row-list GATHER, gated by the ENGINE_VERIFY_DEVICE re-execution.
//
//   (A) values: tape::rref / kernel_basis / factor_basis / solve_multi_rhs
//       under gpu and hybrid, with the verify re-execution on and off, with the
//       session on and off (ENGINE_ELIM_SESSION=0), entrywise against the
//       engine::blk oracle (BlockElim.h), u32 and u64, with R growing (no
//       rank hint) and the pivot-free / dependent shapes;
//   (B) the device row-list GATHER (cuda::gather's row-list form) against
//       the CPU executor entrywise, under verify (every GPU node re-executed);
//   (C) non-vacuity and residency: under gpu the session opened, mirrors were
//       made, planned H2D were skipped and epilogue D2H deferred; with
//       ENGINE_ELIM_SESSION=0 and under cpu no session opened; the session arm
//       moves fewer transfer bytes than the rollback arm on the same input;
//       pageable_copies == 0; nothing session-bound stays cached on the device.
//
// Skips loudly (exit 0) without a usable CUDA device.  Never changes a value:
// every arm is compared with the CPU oracle.
//
// (S24 CL-R2) ENGINE_ELIM_SESSION is a RETIRED lever (RetiredDeviceLevers.h). The rollback
// comparisons -- the (A) arm "gpu session-off (rollback)" and the (C) transfer-volume
// comparison against the rollback arm -- are legacy arms: they run only in an
// OC_ENABLE_LEGACY=ON build (SKIP-ARM lines otherwise, arms "session-off" and "volume-rollback").
// Every other check, and the session arm of the volume case, runs in every build.
#include "EngineElimSessionValidation.h"

#include <cstdlib>
#include <iostream>
#include <string>

#include "../Cuda/CudaCtx.h"
#include "../ModArithmetic/BlockElim.h"
#include "../ModArithmetic/BlockKernels.h"
#include "../ModArithmetic/Kernels.h"
#include "EngineRuntime.h"
#include "EngineTestUtil.h"
#include "Executor.h"
#include "Recorder.h"
#include "TapeElim.h"
#include "../ModArithmetic/LegacyConfig.h"          // (S24 CL-R2) RunLegacyArm
#include "../ModArithmetic/RetiredDeviceLevers.h"   // (S24 CL-R2) the EFFECTIVE line

namespace {

using namespace engine;
using namespace engine::tape;
using namespace engine::tape::test;
namespace cu = engine::cuda;

Checker g;

struct EnvGuard {
    std::string name, old; bool had;
    EnvGuard(const char* n, const char* v) : name(n) {
        const char* o = std::getenv(n); had = o != nullptr; if (had) old = o;
        if (v) setenv(n, v, 1); else unsetenv(n);
    }
    ~EnvGuard() { if (had) setenv(name.c_str(), old.c_str(), 1); else unsetenv(name.c_str()); }
};

struct Arm { DevicePolicy policy; bool verify; bool session; uint32_t threads; bool batch; const char* graph; const char* name; };

std::vector<Arm> arms() {
    return {
        {DevicePolicy::Gpu,    true,  true,  16, true,  "1", "gpu session verify"},
        {DevicePolicy::Gpu,    false, true,  16, true,  "1", "gpu session"},
        {DevicePolicy::Gpu,    false, true,  1,  false, "2", "gpu session threads=1 batch=0 graph=2"},
        {DevicePolicy::Gpu,    false, false, 16, true,  "1", "gpu session-off (rollback)"},
        {DevicePolicy::Hybrid, true,  true,  16, true,  "1", "hybrid session verify"},
        {DevicePolicy::Hybrid, false, true,  4,  true,  "1", "hybrid session threads=4"},
    };
}

template<class Elt>
void elim_case(Executor<Elt>& ex, Rng& rng, const BlockMat<Elt>& A, const std::string& tag) {
    const Elt p = A.p();
    const ModCtx<Elt> ctx(p);
    const blk::BlockRrefResult<Elt> ref = blk::rref<Elt>(A);
    const blk::BlockRrefResult<Elt> got = rref<Elt>(ex, A);   // no rank hint: R grows (release / re-adopt)
    g.check(got.rank == ref.rank && got.pivots == ref.pivots && got.pivot_rows == ref.pivot_rows, tag + " rref rank/pivots/pivot_rows");
    g.same(cview(got.R), cview(ref.R), tag + " rref R");
    const BlockMat<Elt> kt = kernel_basis<Elt>(ex, A);
    g.check(blk::equal<Elt>(kt, blk::kernel_basis<Elt>(A)), tag + " kernel_basis");
    const std::vector<uint32_t> row_dims = A.col_dims();
    const uint32_t a = A.cols();
    if (a == 0) return;
    const uint32_t nr = 1 + rng.below(3), ng = 4 + rng.below(6);
    DenseBuf<Elt> Rl = random_buf<Elt>(rng, a, nr, p, Layout::ColMajor);
    DenseBuf<Elt> G = random_buf<Elt>(rng, a, ng, p, Layout::ColMajor);
    for (uint32_t i = 0; i < a; ++i) {
        G.at(i, 1) = Rl.at(i, 0);
        G.at(i, 3) = ctx.add(G.at(i, 0), ctx.mul(Elt(2 % p), G.at(i, 2)));
    }
    for (uint32_t chunk : {256u, 7u}) {
        const blk::FactorBasisResult<Elt> fr = blk::factor_basis<Elt>(A.pool(), p, row_dims, cview(Rl), cview(G), blk::ElimParams{chunk});
        const blk::FactorBasisResult<Elt> ft = factor_basis<Elt>(ex, A.pool(), p, row_dims, cview(Rl), cview(G), chunk);
        g.check(fr.pivots == ft.pivots && fr.independent_relations == ft.independent_relations, tag + " factor_basis pivots chunk=" + std::to_string(chunk));
        g.check(blk::equal<Elt>(fr.basis, ft.basis) && fr.basis.cols() == ft.basis.cols(), tag + " factor_basis basis");
        g.same(cview(fr.projector), cview(ft.projector), tag + " factor_basis projector");
        const uint32_t ny = 2 + rng.below(3);
        DenseBuf<Elt> coef = random_buf<Elt>(rng, ng, ny, p);
        DenseBuf<Elt> Y(a, ny, Layout::ColMajor);
        kernels::gemm<Elt>(ctx, cview(G), cview(coef), view(Y), kernels::GemmOp::Set);
        const DenseBuf<Elt> sr = blk::solve_multi_rhs<Elt>(A.pool(), p, row_dims, cview(G), cview(Y), blk::ElimParams{chunk});
        const DenseBuf<Elt> st = solve_multi_rhs<Elt>(ex, A.pool(), p, row_dims, cview(G), cview(Y), chunk);
        g.same(cview(sr), cview(st), tag + " solve_multi_rhs chunk=" + std::to_string(chunk));
    }
}

template<class Elt>
void run_values(Elt p, uint64_t seed, const std::string& wtag, bool big) {
    struct Grid { std::vector<uint32_t> rd, cd; uint32_t density; const char* name; };
    std::vector<Grid> grids{
        {{17, 17}, std::vector<uint32_t>(11, 17), 25, "2x11 d=17"},
        {{3, 7, 2}, {5, 1, 6}, 70, "non-uniform"},
        {std::vector<uint32_t>(13, 17), std::vector<uint32_t>(7, 17), 22, "13x7 d=17"},
        {{6}, {6}, 100, "single block"},
    };
    if (big) {
        grids.push_back({{65, 65}, std::vector<uint32_t>(11, 65), 25, "2x11 d=65"});
        grids.push_back({std::vector<uint32_t>(6, 129), std::vector<uint32_t>(7, 129), 22, "6x7 d=129"});
    }
    for (const Arm& arm : arms()) {
        // (S24 CL-R2) the rollback arm sets ENGINE_ELIM_SESSION=0, which an OFF build refuses
        if (!arm.session && !oc::legacy::RunLegacyArm("EngineElimSessionValidation", "session-off")) continue;
        EnvGuard es("ENGINE_ELIM_SESSION", arm.session ? nullptr : "0");
        EnvGuard eg("ENGINE_CUDA_GRAPH", arm.graph);
        PassOptions o;
        o.batch = arm.batch;
        o.policy = arm.policy;
        ScopedPassOptions so(o);
        ThreadPool pool(arm.threads);
        Executor<Elt> ex(&pool);
        ex.set_verify_device(arm.verify);
        const auto t0 = std::chrono::steady_clock::now();
        const DeviceStats before = ex.device_usable() ? ex.device_stats() : DeviceStats();
        uint64_t s = seed;
        for (const Grid& gr : grids)
            for (Shape shape : {Shape::CqLike, Shape::ZqLike, Shape::Generic, Shape::Dependent}) {
                auto bp = std::make_shared<BlockPool<Elt>>();
                Rng rng(s++ ^ static_cast<uint64_t>(p));
                const BlockMat<Elt> A = make_input<Elt>(rng, bp, p, shape, gr.rd, gr.cd, gr.density);
                elim_case<Elt>(ex, rng, A, wtag + " [" + arm.name + "] " + gr.name + " " + shape_name(shape));
            }
        const DeviceStats& after = ex.device_stats();
        const uint64_t sessions = after.sessions - before.sessions;
        if (arm.session) {
            // hybrid's cost model may keep every node of a u32 elimination on the
            // CPU (no mirror needed); the session still opens under it
            g.check(sessions > 0, wtag + " [" + arm.name + "] sessions opened");
            if (arm.policy == DevicePolicy::Gpu)
                g.check(after.session_mirrors > before.session_mirrors, wtag + " [" + arm.name + "] session mirrors made");
            if (arm.policy == DevicePolicy::Gpu && !arm.verify) {
                g.check(after.session_skipped_h2d > before.session_skipped_h2d, wtag + " [" + arm.name + "] planned H2D skipped (residency carried)");
                g.check(after.session_deferred_d2h > before.session_deferred_d2h, wtag + " [" + arm.name + "] epilogue D2H deferred");
            }
        } else {
            g.check(sessions == 0, wtag + " [" + arm.name + "] ENGINE_ELIM_SESSION=0 opens no session");
        }
        // nothing session-bound stays cached: a released executor holds no device bytes
        ex.release_device_state();
        g.check(ex.device_bytes() == 0, wtag + " [" + arm.name + "] device bytes 0 after release");
        std::cout << "  " << wtag << " " << arm.name << ": sessions " << sessions << ", mirrors "
                  << (after.session_mirrors - before.session_mirrors) << ", H2D skipped " << (after.session_skipped_h2d - before.session_skipped_h2d)
                  << ", D2H deferred " << (after.session_deferred_d2h - before.session_deferred_d2h) << ", host syncs "
                  << (after.session_syncs - before.session_syncs) << ", release D2H " << (after.session_release_d2h - before.session_release_d2h)
                  << ", verified nodes " << (after.verified_nodes - before.verified_nodes) << ", " << seconds_since(t0) << " s" << std::endl;
    }
}

// (B) the row-list GATHER on the device vs the CPU executor.
template<class Elt>
void run_row_gather(Elt p, uint64_t seed, const std::string& wtag) {
    const ModCtx<Elt> ctx(p);
    Rng rng(seed);
    for (uint32_t trial = 0; trial < 6; ++trial) {
        const uint32_t ra = 5 + rng.below(60), ca = 3 + rng.below(70), nr = 1 + rng.below(ra + 3);
        const bool col_major = (trial & 1) != 0, accumulate = (trial % 3) == 2, ident = (trial == 4);
        const DenseBuf<Elt> Asrc = random_buf<Elt>(rng, ra, ca, p, col_major ? Layout::ColMajor : Layout::RowMajor);
        std::vector<uint32_t> rows(nr), cols;
        for (uint32_t i = 0; i < nr; ++i) rows[i] = rng.below(ra);   // repeats allowed
        const bool with_cols = (trial % 2) == 0;
        if (with_cols) { const uint32_t k = 1 + rng.below(ca); for (uint32_t t = 0; t < k; ++t) cols.push_back(rng.below(ca)); }
        const uint32_t nc = with_cols ? static_cast<uint32_t>(cols.size()) : ca;
        const Elt alpha = rng.elt<Elt>(p), beta = ident ? Elt(0) : rng.nonzero<Elt>(p);
        const uint32_t row_off = rng.below(4);
        const DenseBuf<Elt> C0 = random_buf<Elt>(rng, nr, nc, p);
        auto run = [&](DevicePolicy pol, bool verify) {
            DenseBuf<Elt> C(C0);
            Recorder<Elt> rec(ctx);
            const BufId bA = rec.input(cview(Asrc));
            const BufId bC = rec.output(view(C));
            const uint32_t rl = rec.index_list(rows);
            const uint32_t cl = with_cols ? rec.index_list(cols) : kNoList;
            const ViewRef av = ident ? ViewRef{kIdentityBuf, 0, ra, 0, ca, false} : rec.whole(bA);
            rec.gather_rows(alpha, beta, av, row_off, rl, cl, rec.whole(bC), accumulate);
            Program<Elt> prog = rec.finish();
            PassOptions o; o.policy = pol;
            compile(prog, o);
            Executor<Elt> ex;
            ex.set_verify_device(verify);
            (void)ex.run(static_cast<const Program<Elt>&>(prog), rec.bindings());
            return std::make_pair(C, prog.stats.gpu_nodes);
        };
        const auto cpu = run(DevicePolicy::Cpu, false);
        const auto gv = run(DevicePolicy::Gpu, true);
        const auto gn = run(DevicePolicy::Gpu, false);
        const std::string tag = wtag + " row-list GATHER trial " + std::to_string(trial);
        g.check(gv.second >= 1 && gn.second >= 1, tag + " assigned to the GPU");
        g.same(cview(gv.first), cview(cpu.first), tag + " gpu (verify) == cpu");
        g.same(cview(gn.first), cview(cpu.first), tag + " gpu == cpu");
    }
}

// (C) transfer volume: the session arm against the rollback arm on one input.
void run_volume() {
    const uint32_t p = 134217757u;
    auto measure = [&](bool session, uint64_t& bytes, uint64_t& h2d, blk::BlockRrefResult<uint32_t>& res) {
        EnvGuard es("ENGINE_ELIM_SESSION", session ? nullptr : "0");
        PassOptions o; o.policy = DevicePolicy::Gpu;
        ScopedPassOptions so(o);
        auto bp = std::make_shared<BlockPool<uint32_t>>();
        Rng rng(0xA19A);
        const BlockMat<uint32_t> A = make_input<uint32_t>(rng, bp, p, Shape::CqLike, std::vector<uint32_t>(13, 129),
                                                          std::vector<uint32_t>(11, 129), 30);
        Executor<uint32_t> ex;
        (void)ex.device_usable();
        const DeviceStats b0 = ex.device_stats();
        res = rref<uint32_t>(ex, A);
        const DeviceStats& b1 = ex.device_stats();
        bytes = (b1.h2d_bytes - b0.h2d_bytes) + (b1.d2h_bytes - b0.d2h_bytes) + (b1.session_release_d2h_bytes - b0.session_release_d2h_bytes);
        h2d = b1.h2d - b0.h2d;
        g.same(cview(res.R), cview(blk::rref<uint32_t>(A).R), std::string("volume case R, session ") + (session ? "on" : "off"));
    };
    cu::CudaCtx::get().reset_transfer_stats();
    uint64_t on = 0, off = 0, h_on = 0, h_off = 0;
    blk::BlockRrefResult<uint32_t> r_on, r_off;
    measure(true, on, h_on, r_on);
    if (oc::legacy::RunLegacyArm("EngineElimSessionValidation", "volume-rollback")) {   // (S24 CL-R2) needs ENGINE_ELIM_SESSION=0
        measure(false, off, h_off, r_off);
        g.check(on < off, "session arm moves fewer bytes than the rollback arm");
        g.check(h_on < h_off, "session arm issues fewer H2D copies than the rollback arm");
    }
    g.check(cu::CudaCtx::get().transfer_stats().pageable_copies == 0, "pageable_copies == 0");
    std::cout << "  transfer volume, rref 1677x1419 (13x11, d=129) under gpu: session " << on / 1048576.0 << " MB in " << h_on << " H2D";
    if (oc::legacy::kLegacyBuild) std::cout << ", rollback " << off / 1048576.0 << " MB in " << h_off << " H2D";
    std::cout << std::endl;
}

// (C) the cpu default never opens a session (and never needs the device).
void run_cpu_default() {
    PassOptions o; o.policy = DevicePolicy::Cpu;
    ScopedPassOptions so(o);
    auto bp = std::make_shared<BlockPool<uint32_t>>();
    Rng rng(0xC9A);
    const BlockMat<uint32_t> A = make_input<uint32_t>(rng, bp, 134217757u, Shape::Generic, {17, 17, 17}, {17, 17, 17, 17}, 60);
    Executor<uint32_t> ex;
    const auto r = rref<uint32_t>(ex, A);
    g.same(cview(r.R), cview(blk::rref<uint32_t>(A).R), "cpu policy R");
    g.check(!ex.elim_session_open() && ex.device_stats().sessions == 0, "cpu policy: no session, no device state");
}

} // namespace

uint32_t EngineElimSessionValidation() {
    const auto t0 = std::chrono::steady_clock::now();
    g = Checker();
    std::cout << "=== EngineElimSessionValidation (slot-18 WP-19a, GPU G1: elimination-session residency + device row-list GATHER, "
                 "entrywise vs engine::blk and the CPU executor, ENGINE_VERIFY_DEVICE arms) ===" << std::endl;
    std::cout << engine::lever::retired_device_lever_line(engine::lever::elim_session_lever()) << std::endl;   // (S24 CL-R2)
    if (!cu::available()) {
        std::cout << "[SKIP] EngineElimSessionValidation: no usable CUDA device (" << cu::unavailable_reason()
                  << ") -- the session is device-only; the cpu default never opens one" << std::endl;
        return 0;
    }
    try {
        run_cpu_default();
        run_values<uint32_t>(134217757u, 0x19A0, "[u32 p=134217757]", true);
        run_values<uint32_t>(7u, 0x19A1, "[u32 p=7]", false);
        run_values<uint32_t>(4294967291u, 0x19A2, "[u32 p=2^32-5]", false);
        run_values<uint64_t>(2305843009213693951ull, 0x19A3, "[u64 p=2^61-1]", false);
        run_row_gather<uint32_t>(134217757u, 0x19A4, "[u32 p=134217757]");
        run_row_gather<uint32_t>(4294967291u, 0x19A5, "[u32 p=2^32-5]");
        run_row_gather<uint64_t>(2305843009213693951ull, 0x19A6, "[u64 p=2^61-1]");
        run_volume();
    } catch (const std::exception& e) {
        std::cout << "  EXCEPTION: " << e.what() << std::endl;
        ++g.fails;
    }
    std::cout << "EngineElimSessionValidation: " << g.checks << " checks, " << g.fails << " failed, " << seconds_since(t0) << " s" << std::endl;
    return g.fails == 0 ? 0 : 1;
}
