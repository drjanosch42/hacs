// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// EngineCoreScalingValidation.cpp -- the performance proof of the core alone
// (Phase 4 deliverable 8): a synthetic Theta-like program (2000 block GEMMs of
// 129 x 129 blocks, 8 distinct pool operands, accumulated into 250 output
// blocks) and one dense 2048^3 GEMM, at ENGINE_THREADS = 1, 4, 16, 32.
// Throughput and speedup are REPORTED; the validation gates only on the
// FNV-64 hashes being identical across thread counts.
#include "EngineCoreScalingValidation.h"

#include <algorithm>
#include <iomanip>
#include <iostream>

#include "EngineTestUtil.h"
#include "Executor.h"
#include "Passes.h"
#include "Recorder.h"
#include "ThreadPool.h"

namespace {

using namespace engine;
using namespace engine::tape;
using namespace engine::tape::test;
using kernels::GemmOp;

Checker g;

struct Row { uint32_t threads; double theta_s, theta_rate, thetac_s, thetac_rate, dense_s, dense_rate; uint64_t h_theta, h_thetac, h_dense; };

double best_of(uint32_t reps, const std::function<double()>& f) {
    double best = 1e300;
    for (uint32_t r = 0; r < reps; ++r) best = std::min(best, f());
    return best;
}

} // namespace

uint32_t EngineCoreScalingValidation() {
    using Elt = uint32_t;
    const auto t0 = std::chrono::steady_clock::now();
    g = Checker();
    std::cout << "=== EngineCoreScalingValidation (core throughput at ENGINE_THREADS = 1, 4, 16, 32; identity gated) ===" << std::endl;
    const Elt p = 134217757u;
    const ModCtx<Elt> ctx(p);
    const uint32_t d = 129, n_ops = 8, n_blocks = 250, terms = 2000, N = 2048;
    const uint32_t reps = std::getenv("ENGINE_SCALING_REPS") ? static_cast<uint32_t>(std::atoi(std::getenv("ENGINE_SCALING_REPS"))) : 3;
    Rng rng(0x5CA1E);
    // Theta-like: Y[b] += rho_{a(i)} . X[b(i)], 2000 terms over 8 operands and 250 blocks
    std::vector<DenseBuf<Elt>> rho;
    for (uint32_t i = 0; i < n_ops; ++i) rho.push_back(random_buf<Elt>(rng, d, d, p));
    const DenseBuf<Elt> X = random_buf<Elt>(rng, n_blocks * d, d, p, Layout::ColMajor);
    const double theta_work = static_cast<double>(terms) * d * d * d;
    const DenseBuf<Elt> DA = random_buf<Elt>(rng, N, N, p), DB = random_buf<Elt>(rng, N, N, p);
    const double dense_work = static_cast<double>(N) * N * N;

    std::vector<Row> rows;
    try {
        for (uint32_t threads : {1u, 4u, 16u, 32u}) {
            ThreadPool pool(threads);
            Executor<Elt> ex(&pool);
            Row row{threads, 0, 0, 0, 0, 0, 0, 0, 0, 0};
            // theta-like, output Theta_flat-style (row-major, block windows row-contiguous)
            // and multi-vector-style (column-major, the L1 layout)
            for (Layout yl : {Layout::RowMajor, Layout::ColMajor}) {
                DenseBuf<Elt> Y(n_blocks * d, d, yl);
                Recorder<Elt> rec(ctx);
                std::vector<BufId> bR;
                for (uint32_t i = 0; i < n_ops; ++i) bR.push_back(rec.input(cview(rho[i]), 100 + i));
                const BufId bX = rec.input(cview(X)), bY = rec.output(view(Y));
                rec.zero(rec.whole(bY));
                for (uint32_t t = 0; t < terms; ++t) {
                    const uint32_t a = (t * 7 + 3) % n_ops, b = (t * 13 + t / n_blocks) % n_blocks;
                    rec.gemm_ref(Elt(0), Elt(1), rec.whole(bR[a]), Elt(0), Elt(1), rec.view(bX, b * d, d, 0, d),
                                 rec.view(bY, b * d, d, 0, d), d, d, d, GemmOp::Add);
                }
                Program<Elt> prog = rec.finish();
                compile(prog, PassOptions());
                const Bindings<Elt> bind = rec.bindings();
                const double secs = best_of(reps, [&] { const RunResult r = ex.run(prog, bind); return r.seconds; });
                if (yl == Layout::RowMajor) { row.theta_s = secs; row.theta_rate = theta_work / secs / 1e9; row.h_theta = hash_view(cview(Y)); }
                else { row.thetac_s = secs; row.thetac_rate = theta_work / secs / 1e9; row.h_thetac = hash_view(cview(Y)); }
                if (threads == 1 && yl == Layout::RowMajor) std::cout << "  theta-like program: " << prog.summary() << std::endl;
            }
            // dense 2048^3
            {
                DenseBuf<Elt> C(N, N);
                Recorder<Elt> rec(ctx);
                const BufId a = rec.input(cview(DA)), b = rec.input(cview(DB)), c = rec.output(view(C));
                rec.gemm(rec.whole(a), rec.whole(b), rec.whole(c), GemmOp::Set);
                Program<Elt> prog = rec.finish();
                compile(prog, PassOptions());
                const Bindings<Elt> bind = rec.bindings();
                row.dense_s = best_of(reps, [&] { const RunResult r = ex.run(prog, bind); return r.seconds; });
                row.dense_rate = dense_work / row.dense_s / 1e9;
                row.h_dense = hash_view(cview(C));
            }
            rows.push_back(row);
        }
    } catch (const std::exception& e) {
        std::cout << "  EXCEPTION: " << e.what() << std::endl;
        ++g.fails;
    }
    std::cout << std::fixed << std::setprecision(3);
    std::cout << "  threads | theta 2000x129^3 row-major out (4.29 G) | theta col-major out (4.29 G)       | dense 2048^3 (8.59 G)" << std::endl;
    std::cout << "          |     s    Gmodmul/s  speedup             |     s    Gmodmul/s  speedup        |     s    Gmodmul/s  speedup" << std::endl;
    for (const Row& r : rows) {
        std::cout << "  " << std::setw(7) << r.threads
                  << " | " << std::setw(6) << r.theta_s << "  " << std::setw(8) << r.theta_rate << "  " << std::setw(7) << (rows[0].theta_s / r.theta_s) << "x           "
                  << " | " << std::setw(6) << r.thetac_s << "  " << std::setw(8) << r.thetac_rate << "  " << std::setw(7) << (rows[0].thetac_s / r.thetac_s) << "x      "
                  << " | " << std::setw(6) << r.dense_s << "  " << std::setw(8) << r.dense_rate << "  " << std::setw(7) << (rows[0].dense_s / r.dense_s) << "x" << std::endl;
        g.check(r.h_theta == rows[0].h_theta, "theta hash identical at " + std::to_string(r.threads) + " threads");
        g.check(r.h_thetac == rows[0].h_thetac && r.h_thetac == rows[0].h_theta, "theta (col-major out) hash identical at " + std::to_string(r.threads) + " threads");
        g.check(r.h_dense == rows[0].h_dense, "dense hash identical at " + std::to_string(r.threads) + " threads");
    }
    if (rows.size() == 4) {
        std::cout << "  SCALING theta16=" << rows[0].theta_s / rows[2].theta_s << "x theta32=" << rows[0].theta_s / rows[3].theta_s
                  << "x thetacol16=" << rows[0].thetac_s / rows[2].thetac_s << "x dense16=" << rows[0].dense_s / rows[2].dense_s << "x dense32=" << rows[0].dense_s / rows[3].dense_s
                  << "x dense_rate16=" << rows[2].dense_rate << " dense_rate32=" << rows[3].dense_rate
                  << " (target: >= 8x at 16 threads on the batched program; dense near ~512 Gmodmul/s, doc 04 Sec. 1.5)" << std::endl;
    }
    std::cout << "EngineCoreScalingValidation: " << g.checks << " checks, " << g.fails << " failed, " << seconds_since(t0) << " s" << std::endl;
    return g.fails == 0 ? 0 : 1;
}
