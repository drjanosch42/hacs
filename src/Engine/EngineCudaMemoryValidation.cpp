// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// EngineCudaMemoryValidation.cpp -- the tape's device memory contract
// (matrix-engine Phase 5 Stage B, plan Phase 5 "EngineCudaMemoryValidation").
//
//   * the pinned staging pool never exceeds its cap, even when one input is
//     larger than the whole pool (a 144 MB GEMM operand through a 128 MB pool)
//   * pageable_copies stays 0 across every run (no un-staged copy, ever)
//   * the device arena returns to its baseline after the executor releases
//     its device state (mirrors, graphs, batch tables, workspaces); the LRU
//     of device programs is bounded
//   * bit-identity is asserted on the way (every program's output equals the
//     CPU result)
//   * (H, slot-14 WP-3) release_device_caches() returns every idle arena slab
//     and the pinned pool to the driver: the same kind of work run twice in
//     one process (shifted shapes, so no exact-size reuse) has a bounded
//     high-water after a release, and grows without one (the counterfactual)
#include "EngineCudaMemoryValidation.h"

#include <cstdlib>
#include <iostream>
#include <memory>
#include <string>

#include "../Cuda/CudaCtx.h"
#include "../ModArithmetic/BlockKernels.h"
#include "../ModArithmetic/Kernels.h"
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

} // namespace

uint32_t EngineCudaMemoryValidation() {
    using Elt = uint32_t;
    const auto t0 = std::chrono::steady_clock::now();
    g = Checker();
    std::cout << "=== EngineCudaMemoryValidation (pinned pool <= cap, zero pageable copies, arena back to baseline) ===" << std::endl;
    if (!cu::available()) {
        std::cout << "[SKIP] EngineCudaMemoryValidation: no usable CUDA device (" << cu::unavailable_reason() << ")" << std::endl;
        return 0;
    }
    try {
        EnvGuard d("ENGINE_DEVICE", "gpu");
        cu::CudaCtx& cx = cu::CudaCtx::get();
        const Elt p = 134217757u;
        const ModCtx<Elt> ctx(p);
        PassOptions opts; opts.policy = DevicePolicy::Gpu;
        cx.reset_transfer_stats();
        const cu::ArenaStats a0 = cx.arena_stats();
        const cu::PinnedPoolStats p0 = cx.pinned_stats();
        std::cout << "  baseline: arena live " << a0.live_bytes << " B (reserved " << a0.reserved_bytes << " B), pinned "
                  << p0.allocated_bytes << " B of cap " << p0.cap_bytes << " B" << std::endl;
        Rng rng(0x3E3);
        size_t bytes_live_before_release = 0;
        {
            std::unique_ptr<Executor<Elt>> ex(new Executor<Elt>());
            ex->set_verify_device(true);
            g.check(ex->device_usable(), "executor has a device backend");
            // 1. a 6000 x 6000 input (144 MB > the 128 MB pinned cap) times 6000 x 64, twice (replay)
            {
                const uint32_t N = 6000, w = 64;
                const DenseBuf<Elt> A = random_buf<Elt>(rng, N, N, p), B = random_buf<Elt>(rng, N, w, p);
                DenseBuf<Elt> C(N, w), Cref(N, w);
                kernels::gemm<Elt>(ctx, cview(A), cview(B), view(Cref), GemmOp::Set);
                Recorder<Elt> rec(ctx);
                const BufId a = rec.input(cview(A)), b = rec.input(cview(B)), c = rec.output(view(C));
                rec.gemm(rec.whole(a), rec.whole(b), rec.whole(c), GemmOp::Set);
                Program<Elt> prog = rec.finish();
                for (int rep = 0; rep < 2; ++rep) {
                    C.fill(0);
                    ex->run(prog, rec.bindings(), opts);
                    g.same(cview(C), cview(Cref), "6000x6000x64 GEMM through a 144 MB upload, run " + std::to_string(rep));
                }
                const cu::PinnedPoolStats ps = cx.pinned_stats();
                g.check(ps.peak_bytes <= ps.cap_bytes, "pinned peak " + std::to_string(ps.peak_bytes) + " <= cap " + std::to_string(ps.cap_bytes));
                g.check(ex->device_stats().h2d_bytes >= static_cast<uint64_t>(N) * N * sizeof(Elt), "the 144 MB operand went through the staged upload");
                std::cout << "  144 MB operand: pinned peak " << ps.peak_bytes << " B, waits " << ps.waits << ", h2d "
                          << ex->device_stats().h2d_bytes << " B, d2h " << ex->device_stats().d2h_bytes << " B" << std::endl;
            }
            // 2. a Theta-like batched program with a 250-block column-major multi-vector, replayed (graph)
            {
                const uint32_t dd = 129, nb = 250, terms = 2000;
                std::vector<DenseBuf<Elt>> rho;
                for (uint32_t i = 0; i < 8; ++i) rho.push_back(random_buf<Elt>(rng, dd, dd, p));
                const DenseBuf<Elt> X = random_buf<Elt>(rng, nb * dd, dd, p, Layout::ColMajor);
                DenseBuf<Elt> Y(nb * dd, dd), Yref(nb * dd, dd);
                Recorder<Elt> rec(ctx);
                std::vector<BufId> bR;
                for (uint32_t i = 0; i < 8; ++i) bR.push_back(rec.input(cview(rho[i]), 100 + i));
                const BufId bX = rec.input(cview(X)), bY = rec.output(view(Y));
                rec.zero(rec.whole(bY));
                for (uint32_t t = 0; t < terms; ++t) {
                    const uint32_t a = (t * 7 + 3) % 8, b = (t * 13 + t / nb) % nb;
                    rec.gemm_ref(Elt(0), Elt(1), rec.whole(bR[a]), Elt(0), Elt(1), rec.view(bX, b * dd, dd, 0, dd),
                                 rec.view(bY, b * dd, dd, 0, dd), dd, dd, dd, GemmOp::Add);
                }
                Program<Elt> prog = rec.finish();
                {
                    PassOptions cpu; Executor<Elt> exc;
                    Program<Elt> pc = prog;   // same recording, CPU arm
                    Bindings<Elt> bc = rec.bindings(); bc.outputs[0] = view(Yref);
                    exc.run(pc, bc, cpu);
                }
                for (int rep = 0; rep < 3; ++rep) {
                    Y.fill(0);
                    ex->run(prog, rec.bindings(), opts);
                    g.same(cview(Y), cview(Yref), "theta-like 2000 x 129^3 on the device, run " + std::to_string(rep));
                }
                g.check(ex->device_stats().graph_launches >= 2, "replays used the captured graph");
            }
            // 3. the elimination driver (many one-shot programs -> LRU eviction) on a 13x7 d=129 Cq matrix
            {
                auto bpool = std::make_shared<BlockPool<Elt>>();
                const BlockMat<Elt> A = make_input<Elt>(rng, bpool, p, Shape::CqLike, std::vector<uint32_t>(13, 129), std::vector<uint32_t>(7, 129), 18, 8);
                const blk::BlockRrefResult<Elt> ref = blk::rref<Elt>(A);
                const blk::BlockRrefResult<Elt> got = rref<Elt>(*ex, A);
                g.check(got.rank == ref.rank && got.pivots == ref.pivots, "elimination rank / pivots");
                g.same(cview(got.R), cview(ref.R), "elimination R on the device");
                const DeviceStats& ds = ex->device_stats();
                g.check(ds.programs_built > ds.programs_evicted, "device program cache holds live programs");
                std::cout << "  device programs built " << ds.programs_built << ", evicted " << ds.programs_evicted
                          << ", gpu nodes " << ds.gpu_nodes << ", verified " << ds.verified_nodes << std::endl;
            }
            const cu::TransferStats ts = cx.transfer_stats();
            g.check(ts.pageable_copies == 0, "pageable_copies == 0 across the run (" + std::to_string(ts.pageable_copies) + ")");
            bytes_live_before_release = cx.arena_stats().live_bytes;
            g.check(ex->device_bytes() > 0 && bytes_live_before_release > a0.live_bytes, "device state is live before release");
            ex->release_device_state();
            g.check(ex->device_bytes() == 0, "device_bytes() == 0 after release_device_state()");
            const cu::ArenaStats a1 = cx.arena_stats();
            g.check(a1.live_bytes == a0.live_bytes, "arena live bytes back to baseline after release (" +
                    std::to_string(a1.live_bytes) + " vs " + std::to_string(a0.live_bytes) + ")");
        }
        const cu::ArenaStats a2 = cx.arena_stats();
        g.check(a2.live_bytes == a0.live_bytes, "arena live bytes back to baseline after executor teardown");
        const cu::PinnedPoolStats p1 = cx.pinned_stats();
        std::cout << "  after teardown: arena live " << a2.live_bytes << " B (peak live " << a2.peak_live_bytes << " B, before release "
                  << bytes_live_before_release << " B), pinned " << p1.allocated_bytes << " B, peak " << p1.peak_bytes << " B" << std::endl;

        // (H) slot-14 WP-3: the release returns the arena's cached slabs to the
        // driver, so the same kind of work run twice in one process has a
        // bounded high-water.  The workload W(s) is two GEMM programs whose
        // buffer sizes depend on s, so W(s) and W(s') for s != s' never share
        // an exact-size free-list entry -- the pattern of consecutive
        // validations that made `ENGINE_DEVICE=gpu --all` climb to the card's
        // capacity ([C4] Sec. 2, [C7] headline 3).
        {
            auto workload = [&](uint32_t s, const std::string& tag) {
                std::unique_ptr<Executor<Elt>> exw(new Executor<Elt>());
                for (const uint32_t N0 : {3000u, 4000u}) {
                    const uint32_t N = N0 + 8 * s, w = 64 + s;
                    const DenseBuf<Elt> A = random_buf<Elt>(rng, N, N, p), B = random_buf<Elt>(rng, N, w, p);
                    DenseBuf<Elt> C(N, w), Cref(N, w);
                    kernels::gemm<Elt>(ctx, cview(A), cview(B), view(Cref), GemmOp::Set);
                    Recorder<Elt> rec(ctx);
                    const BufId a = rec.input(cview(A)), b = rec.input(cview(B)), c = rec.output(view(C));
                    rec.gemm(rec.whole(a), rec.whole(b), rec.whole(c), GemmOp::Set);
                    Program<Elt> prog = rec.finish();
                    exw->run(prog, rec.bindings(), opts);
                    g.same(cview(C), cview(Cref), "(H) " + tag + " GEMM " + std::to_string(N) + "x" + std::to_string(N) + "x" + std::to_string(w));
                }
            };
            const cu::ReleaseStats r0 = cu::release_device_caches();
            const cu::ArenaStats h0 = cx.arena_stats();
            g.check(cu::context_created(), "(H) the context exists (release_device_caches acts on it)");
            g.check(h0.live_bytes != 0 || h0.reserved_bytes == 0,
                    "(H) nothing live -> the release leaves nothing reserved (" + std::to_string(h0.reserved_bytes) + " B)");
            g.check(cx.pinned_stats().allocated_bytes == 0, "(H) the pinned staging pool is freed by the release");
            workload(1, "run 1");
            const size_t high1 = cx.arena_stats().reserved_bytes;
            g.check(high1 > h0.reserved_bytes, "(H) run 1 reserved device memory");
            // counterfactual: the same kind of work again WITHOUT a release grows the arena
            workload(2, "no-release");
            const size_t grown = cx.arena_stats().reserved_bytes;
            g.check(grown > high1, "(H) without a release the arena grows (" + std::to_string(high1 >> 20) + " -> " +
                    std::to_string(grown >> 20) + " MiB): the defect the release answers");
            const cu::ReleaseStats r1 = cu::release_device_caches();
            const cu::ArenaStats h1 = cx.arena_stats();
            g.check(r1.arena_bytes_freed >= grown - h0.reserved_bytes && h1.reserved_bytes == h0.reserved_bytes,
                    "(H) the release returns every slab the two runs reserved (" + std::to_string(r1.arena_bytes_freed >> 20) + " MiB freed)");
            g.check(h1.live_bytes == h0.live_bytes, "(H) no live block was touched");
            workload(3, "run 2");
            const size_t high2 = cx.arena_stats().reserved_bytes;
            const size_t slack = size_t(64) << 20;   // one arena slab
            g.check(high2 <= high1 + slack, "(H) run 2's high-water " + std::to_string(high2 >> 20) + " MiB <= run 1's " +
                    std::to_string(high1 >> 20) + " MiB + one slab");
            (void)cu::release_device_caches();
            const cu::DeviceInfo di = cu::device_info();
            std::cout << "  (H) arena reserved: base " << (h0.reserved_bytes >> 20) << " MiB, run 1 " << (high1 >> 20)
                      << " MiB, no-release run " << (grown >> 20) << " MiB, after release " << (h1.reserved_bytes >> 20)
                      << " MiB, run 2 " << (high2 >> 20) << " MiB; released " << ((r0.arena_bytes_freed + r1.arena_bytes_freed) >> 20)
                      << " MiB in " << (r0.slabs_freed + r1.slabs_freed) << " slabs; driver free " << (di.free_mem >> 20)
                      << " of " << (di.total_mem >> 20) << " MiB" << std::endl;
        }
    } catch (const std::exception& e) {
        std::cout << "  EXCEPTION: " << e.what() << std::endl;
        ++g.fails;
    }
    std::cout << "EngineCudaMemoryValidation: " << g.checks << " checks, " << g.fails << " failed, " << seconds_since(t0) << " s" << std::endl;
    return g.fails == 0 ? 0 : 1;
}
