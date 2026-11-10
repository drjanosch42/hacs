// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// EngineDeviceTrapValidation.cpp -- the "unblocked-kernel trap" evidence
// (matrix-engine Phase 5 Stage B, plan Phase 5 / doc 04 Sec. 6.3, risk R6).
//
// Measures, on this machine, the same tape programs under ENGINE_DEVICE =
// cpu / gpu / hybrid (wall time INCLUDING every transfer, first run and
// replay):
//   A. TapeElim::rref on a 1677 x 1419 rank-258 block matrix (13 x 11, d = 129)
//   B. TapeElim::rref on a dense 4096^2 full-rank block matrix (16 x 16, d = 256)
//   C. the Theta-like batched program 2000 x 129^3 (8 operands, 250 blocks)
//   D. one dense 4096^3 GEMM -- which MUST k-block into >= 4 chunks at
//      ell = 134217757 (k_max = 1023): the chunk count is reported, and the
//      deliberately unblocked variant (4096 products in one accumulator) is
//      shown to trip ModContextValidation's k_max assertion and to overflow
//   E. the Phase 5 exit-criterion batch: d = 100, 256 instances, GPU vs the
//      CPU single-thread rate
// and prints the exit-criterion verdicts (blocked GEMM >= 2 TmodMAC/s at 4096^3
// through the tape; batched d = 100 >= 20x CPU-1).  Gated on identity only:
// every policy must produce the same FNV-64 hashes.  Slow tier.
#include "EngineDeviceTrapValidation.h"

#include <algorithm>
#include <cstdlib>
#include <functional>
#include <iomanip>
#include <iostream>
#include <string>

#include "../Cuda/CudaCtx.h"
#include "../ModArithmetic/BlockKernels.h"
#include "../ModArithmetic/Kernels.h"
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
using Elt = uint32_t;

Checker g;

struct EnvGuard {
    std::string name, old; bool had;
    EnvGuard(const char* n, const char* v) : name(n) {
        const char* o = std::getenv(n); had = o != nullptr; if (had) old = o;
        setenv(n, v, 1);
    }
    ~EnvGuard() { if (had) setenv(name.c_str(), old.c_str(), 1); else unsetenv(name.c_str()); }
};

const char* kPolicies[3] = {"cpu", "gpu", "hybrid"};
DevicePolicy policy_of(const char* s) {
    return std::string(s) == "gpu" ? DevicePolicy::Gpu : std::string(s) == "hybrid" ? DevicePolicy::Hybrid : DevicePolicy::Cpu;
}

uint32_t reps() {
    const char* e = std::getenv("ENGINE_TRAP_REPS");
    const int v = e ? std::atoi(e) : 0;
    return v <= 0 ? 3u : static_cast<uint32_t>(v);
}

double best_of(uint32_t n, const std::function<double()>& f) {
    double best = 1e300;
    for (uint32_t r = 0; r < n; ++r) best = std::min(best, f());
    return best;
}

struct Row { std::string what; double t[3]; uint64_t h[3]; uint64_t gpu_nodes[3]; double xfer_mb[3]; };
std::vector<Row> rows;

void print_rows(const std::string& title) {
    std::cout << "  " << title << std::endl;
    std::cout << "  " << std::left << std::setw(44) << "case" << " |     cpu      |     gpu      |    hybrid    | gpu nodes (gpu/hybrid), MB transferred (gpu/hybrid)" << std::endl;
    for (const Row& r : rows) {
        std::cout << "  " << std::left << std::setw(44) << r.what << " |";
        for (int i = 0; i < 3; ++i) std::cout << " " << std::right << std::setw(9) << std::fixed << std::setprecision(3) << (r.t[i] * 1e3) << " ms |";
        std::cout << " " << r.gpu_nodes[1] << " / " << r.gpu_nodes[2] << ", " << std::setprecision(1) << r.xfer_mb[1] << " / " << r.xfer_mb[2] << std::endl;
        g.check(r.h[1] == r.h[0] && r.h[2] == r.h[0], r.what + ": hashes identical under cpu / gpu / hybrid");
    }
    rows.clear();
}

BlockMat<Elt> block_matrix_from_dense(std::shared_ptr<BlockPool<Elt>> pool, const DenseBuf<Elt>& M, uint32_t d, Elt p) {
    const uint32_t nbr = (M.rows + d - 1) / d, nbc = (M.cols + d - 1) / d;
    std::vector<uint32_t> rd(nbr, d), cd(nbc, d);
    rd.back() = M.rows - d * (nbr - 1); cd.back() = M.cols - d * (nbc - 1);
    BlockBuilder<Elt> b(pool, rd, cd, p);
    for (uint32_t br = 0; br < nbr; ++br)
        for (uint32_t bc = 0; bc < nbc; ++bc) {
            DenseBuf<Elt> blk(rd[br], cd[bc]);
            kernels::copy<Elt>(DenseView<const Elt>(M, br * d, rd[br], bc * d, cd[bc], false), view(blk));
            b.set(br, bc, BlockRef<Elt>::dense(pool->intern(std::move(blk))));
        }
    return b.finalize();
}

uint64_t hash_rref(const blk::BlockRrefResult<Elt>& r) {
    uint64_t h = fnv_mix(fnv_init(), r.rank);
    for (uint32_t c : r.pivots) h = fnv_mix(h, c);
    return fnv_mix(h, hash_view(cview(r.R)));
}

// A. / B. elimination
void elimination_case(const std::string& what, const BlockMat<Elt>& A) {
    Row row{what, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}};
    for (int i = 0; i < 3; ++i) {
        EnvGuard d("ENGINE_DEVICE", kPolicies[i]);
        Executor<Elt> ex;
        row.t[i] = best_of(reps(), [&] {
            const auto t0 = std::chrono::steady_clock::now();
            const blk::BlockRrefResult<Elt> r = rref<Elt>(ex, A);
            const double s = seconds_since(t0);
            row.h[i] = hash_rref(r);
            return s;
        });
        row.gpu_nodes[i] = ex.device_stats().gpu_nodes / reps();
        row.xfer_mb[i] = static_cast<double>(ex.device_stats().h2d_bytes + ex.device_stats().d2h_bytes) / reps() / 1e6;
    }
    rows.push_back(row);
}

// C. / E. Theta-like batched programs; first run (direct launches) and replay (graph)
void theta_case(const std::string& what, uint32_t d, uint32_t n_ops, uint32_t n_blocks, uint32_t terms, Elt p, uint32_t cpu_threads,
                double* out_gpu_replay = nullptr, double* out_cpu = nullptr, double* out_work = nullptr) {
    const ModCtx<Elt> ctx(p);
    Rng rng(0x7E7A + d);
    std::vector<DenseBuf<Elt>> rho;
    for (uint32_t i = 0; i < n_ops; ++i) rho.push_back(random_buf<Elt>(rng, d, d, p));
    const DenseBuf<Elt> X = random_buf<Elt>(rng, n_blocks * d, d, p, Layout::ColMajor);
    const double work = static_cast<double>(terms) * d * d * d;
    Row first{what + " (first run)", {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}};
    Row replay{what + " (replay)", {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}};
    for (int i = 0; i < 3; ++i) {
        ThreadPool pool(cpu_threads);
        Executor<Elt> ex(&pool);
        DenseBuf<Elt> Y(n_blocks * d, d);
        Recorder<Elt> rec(ctx);
        std::vector<BufId> bR;
        for (uint32_t k = 0; k < n_ops; ++k) bR.push_back(rec.input(cview(rho[k]), 100 + k));
        const BufId bX = rec.input(cview(X)), bY = rec.output(view(Y));
        rec.zero(rec.whole(bY));
        for (uint32_t t = 0; t < terms; ++t) {
            const uint32_t a = (t * 7 + 3) % n_ops, b = (terms >= n_blocks) ? (t * 13 + t / n_blocks) % n_blocks : t % n_blocks;
            rec.gemm_ref(Elt(0), Elt(1), rec.whole(bR[a]), Elt(0), Elt(1), rec.view(bX, b * d, d, 0, d),
                         rec.view(bY, b * d, d, 0, d), d, d, d, GemmOp::Add);
        }
        Program<Elt> prog = rec.finish();
        PassOptions o; o.policy = policy_of(kPolicies[i]);
        compile(prog, o);
        const Bindings<Elt> bind = rec.bindings();
        {
            const auto t0 = std::chrono::steady_clock::now();
            ex.run(prog, bind);
            first.t[i] = seconds_since(t0);
            first.h[i] = hash_view(cview(Y));
            first.gpu_nodes[i] = ex.device_stats().gpu_nodes;
            first.xfer_mb[i] = static_cast<double>(ex.device_stats().h2d_bytes + ex.device_stats().d2h_bytes) / 1e6;
        }
        const uint64_t n0 = ex.device_stats().gpu_nodes, b0 = ex.device_stats().h2d_bytes + ex.device_stats().d2h_bytes;
        uint32_t k = 0;
        replay.t[i] = best_of(reps(), [&] {
            const auto t0 = std::chrono::steady_clock::now();
            ex.run(prog, bind);
            const double s = seconds_since(t0);
            ++k;
            return s;
        });
        replay.h[i] = hash_view(cview(Y));
        replay.gpu_nodes[i] = (ex.device_stats().gpu_nodes - n0) / k;
        replay.xfer_mb[i] = static_cast<double>(ex.device_stats().h2d_bytes + ex.device_stats().d2h_bytes - b0) / k / 1e6;
        if (i == 0 && cpu_threads == 1) std::cout << "    " << what << " program: " << prog.summary() << std::endl;
        if (i == 1 && std::getenv("ENGINE_TAPE_TRACE")) {
            const DeviceStats& ds = ex.device_stats();
            std::cout << "    gpu arm: launch " << ds.launch_seconds * 1e3 << " ms, wait " << ds.wait_seconds * 1e3 << " ms, transfers "
                      << ds.transfer_seconds * 1e3 << " ms over " << ds.runs << " runs; graph launches " << ds.graph_launches << std::endl;
        }
    }
    if (out_gpu_replay) *out_gpu_replay = replay.t[1];
    if (out_cpu) *out_cpu = replay.t[0];
    if (out_work) *out_work = work;
    rows.push_back(first); rows.push_back(replay);
}

// D. one dense GEMM
void dense_case(const std::string& what, uint32_t N, Elt p, double* out_gpu, double* out_hybrid) {
    const ModCtx<Elt> ctx(p);
    Rng rng(0xDE45E);
    const DenseBuf<Elt> A = random_buf<Elt>(rng, N, N, p), B = random_buf<Elt>(rng, N, N, p);
    Row first{what + " (first run)", {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}};
    Row replay{what + " (replay)", {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}};
    for (int i = 0; i < 3; ++i) {
        Executor<Elt> ex;
        DenseBuf<Elt> C(N, N);
        Recorder<Elt> rec(ctx);
        const BufId a = rec.input(cview(A)), b = rec.input(cview(B)), c = rec.output(view(C));
        rec.gemm(rec.whole(a), rec.whole(b), rec.whole(c), GemmOp::Set);
        Program<Elt> prog = rec.finish();
        PassOptions o; o.policy = policy_of(kPolicies[i]);
        compile(prog, o);
        const Bindings<Elt> bind = rec.bindings();
        {
            const auto t0 = std::chrono::steady_clock::now();
            ex.run(prog, bind);
            first.t[i] = seconds_since(t0);
            first.h[i] = hash_view(cview(C));
            first.gpu_nodes[i] = ex.device_stats().gpu_nodes;
            first.xfer_mb[i] = static_cast<double>(ex.device_stats().h2d_bytes + ex.device_stats().d2h_bytes) / 1e6;
        }
        const uint64_t n0 = ex.device_stats().gpu_nodes, b0 = ex.device_stats().h2d_bytes + ex.device_stats().d2h_bytes;
        uint32_t k = 0;
        replay.t[i] = best_of(reps(), [&] {
            const auto t0 = std::chrono::steady_clock::now();
            ex.run(prog, bind);
            ++k;
            return seconds_since(t0);
        });
        replay.h[i] = hash_view(cview(C));
        replay.gpu_nodes[i] = (ex.device_stats().gpu_nodes - n0) / k;
        replay.xfer_mb[i] = static_cast<double>(ex.device_stats().h2d_bytes + ex.device_stats().d2h_bytes - b0) / k / 1e6;
        if (i == 1) {
            const DeviceStats& ds = ex.device_stats();
            std::cout << "    gpu arm 4096^3: per run transfers " << ds.transfer_seconds / ds.runs * 1e3 << " ms host-side, wait "
                      << ds.wait_seconds / ds.runs * 1e3 << " ms, launch " << ds.launch_seconds / ds.runs * 1e3 << " ms" << std::endl;
        }
    }
    if (out_gpu) *out_gpu = replay.t[1];
    if (out_hybrid) *out_hybrid = replay.t[2];
    rows.push_back(first); rows.push_back(replay);
}

} // namespace

uint32_t EngineDeviceTrapValidation() {
    const auto t0 = std::chrono::steady_clock::now();
    g = Checker();
    std::cout << "=== EngineDeviceTrapValidation (cpu vs gpu vs hybrid on the tape: the unblocked-kernel trap evidence; identity gated) ===" << std::endl;
    if (!cu::available()) {
        std::cout << "[SKIP] EngineDeviceTrapValidation: no usable CUDA device (" << cu::unavailable_reason() << ")" << std::endl;
        return 0;
    }
    const Elt p = 134217757u;
    const ModCtx<Elt> ctx(p);
    try {
        const cu::DeviceInfo di = cu::device_info();
        std::cout << "  device: " << di.name << ", pool " << ThreadPool::global().size() << " threads, best of " << reps() << std::endl;
        // ---- D. k-blocking evidence first (cheap, and the premise of everything below) ----
        {
            const uint64_t kb = std::min<uint64_t>(ctx.k_max, 256);
            const uint32_t chunks = static_cast<uint32_t>((4096 + kb - 1) / kb);
            std::cout << "  k-blocking at ell = " << p << ": k_max = " << ctx.k_max << ", device kernel flush interval kb = " << kb
                      << ", a 4096-deep dot product is " << chunks << " chunks (>= 4 required)" << std::endl;
            g.check(4096 > ctx.k_max, "4096 > k_max: the 4096^3 GEMM cannot be accumulated in one uint64");
            g.check(chunks >= 4, "4096^3 k-blocks into >= 4 chunks");
            // the unblocked variant: ModContextValidation's bound assertion fires, and the sum overflows
            std::vector<Elt> a(4096, p - 1), b(4096, p - 1);
            bool threw = false;
            try { (void)kernels::dot_delayed<Elt>(ctx, a.data(), b.data(), 4096); } catch (const std::logic_error&) { threw = true; }
            g.check(threw, "dot_delayed over 4096 > k_max products trips the ENGINE_CHECK (the unblocked variant is rejected)");
            uint64_t wrapped = 0;
            for (uint32_t i = 0; i < 4096; ++i) wrapped += static_cast<uint64_t>(p - 1) * (p - 1);   // silently wraps mod 2^64
            const Elt exact = ctx.mul(static_cast<Elt>(4096 % p), ctx.mul(p - 1, p - 1));
            g.check(ctx.reduce_u64(wrapped) != exact, "an unblocked uint64 accumulation of 4096 products is WRONG (wrapped mod 2^64)");
            std::cout << "  unblocked variant: dot_delayed(k = 4096) " << (threw ? "throws" : "DID NOT THROW") << "; wrapped accumulator gives "
                      << ctx.reduce_u64(wrapped) << ", exact is " << exact << std::endl;
        }
        // ---- A. 1677 x 1419 rank 258 ----
        {
            Rng rng(0x1677);
            auto pool = std::make_shared<BlockPool<Elt>>();
            const DenseBuf<Elt> L = random_buf<Elt>(rng, 1677, 258, p), U = random_buf<Elt>(rng, 258, 1419, p);
            DenseBuf<Elt> M(1677, 1419);
            kernels::gemm<Elt>(ctx, cview(L), cview(U), view(M), GemmOp::Set);
            const BlockMat<Elt> A = block_matrix_from_dense(pool, M, 129, p);
            elimination_case("A. rref 1677x1419 rank 258 (13x11, d=129)", A);
        }
        // ---- B. dense 4096^2 full rank ----
        {
            Rng rng(0x4096);
            auto pool = std::make_shared<BlockPool<Elt>>();
            const DenseBuf<Elt> M = random_buf<Elt>(rng, 4096, 4096, p);
            const BlockMat<Elt> A = block_matrix_from_dense(pool, M, 256, p);
            elimination_case("B. rref dense 4096^2 full rank (16x16, d=256)", A);
        }
        print_rows("elimination (TapeElim::rref; wall incl. transfers; ENGINE_DEVICE from the environment)");
        // ---- C. Theta-like 2000 x 129^3 ----
        theta_case("C. theta-like 2000 x 129^3 (8 ops, 250 blocks)", 129, 8, 250, 2000, p, ThreadPool::global().size());
        // ---- D. dense 4096^3 ----
        double gpu4096 = 0, hyb4096 = 0;
        dense_case("D. dense GEMM 4096^3", 4096, p, &gpu4096, &hyb4096);
        print_rows("programs (wall incl. transfers; first run = direct launches, replay = captured graph)");
        // ---- E. exit criterion batch: d = 100, 256 instances, vs CPU single thread ----
        double gpu_e = 0, cpu1_e = 0, work_e = 0, gpu_e2 = 0, cpu1_e2 = 0, work_e2 = 0;
        theta_case("E1. batched d=100 x 256 inst, 256 X / 256 Y blocks", 100, 8, 256, 256, p, 1, &gpu_e, &cpu1_e, &work_e);
        theta_case("E2. batched d=100 x 256 inst, 32 X / 32 Y blocks", 100, 8, 32, 256, p, 1, &gpu_e2, &cpu1_e2, &work_e2);
        print_rows("exit-criterion batches (cpu column = ONE thread; E1 moves 20 MB per run, E2 2.6 MB)");
        // ---- verdicts ----
        const double work4096 = 4096.0 * 4096.0 * 4096.0;
        const double rate_gpu = work4096 / gpu4096, rate_hyb = work4096 / hyb4096;
        std::cout << std::fixed << std::setprecision(2);
        std::cout << "  EXIT blocked GEMM 4096^3 through the tape (gpu, replay, incl. 192 MB of transfers): " << rate_gpu / 1e12
                  << " TmodMAC/s (hybrid " << rate_hyb / 1e12 << ") -- target >= 2: " << (rate_gpu >= 2e12 ? "PASS" : "FAIL") << std::endl;
        const double ratio = cpu1_e / gpu_e, ratio2 = cpu1_e2 / gpu_e2;
        std::cout << "  EXIT batched d=100 x 256 through the tape, E1 (256 distinct in/out blocks, transfer-bound): gpu " << gpu_e * 1e3
                  << " ms vs CPU-1 " << cpu1_e * 1e3 << " ms = " << ratio << "x (gpu " << work_e / gpu_e / 1e9 << " G, cpu-1 "
                  << work_e / cpu1_e / 1e9 << " Gmodmul/s) -- target >= 20x: " << (ratio >= 20.0 ? "PASS" : "FAIL") << std::endl;
        std::cout << "  EXIT batched d=100 x 256 through the tape, E2 (32 in/out blocks, the Theta shape): gpu " << gpu_e2 * 1e3
                  << " ms vs CPU-1 " << cpu1_e2 * 1e3 << " ms = " << ratio2 << "x (gpu " << work_e2 / gpu_e2 / 1e9 << " G, cpu-1 "
                  << work_e2 / cpu1_e2 / 1e9 << " Gmodmul/s) -- target >= 20x: " << (ratio2 >= 20.0 ? "PASS" : "FAIL") << std::endl;
    } catch (const std::exception& e) {
        std::cout << "  EXCEPTION: " << e.what() << std::endl;
        ++g.fails;
    }
    std::cout << "EngineDeviceTrapValidation: " << g.checks << " checks, " << g.fails << " failed, " << seconds_since(t0) << " s" << std::endl;
    return g.fails == 0 ? 0 : 1;
}
