// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// hecke-extensions slot 23, PERF-P7 -- GpuLeanSessionValidation (see the header).

#include "GpuLeanSessionValidation.h"

#include "EquivariantCohomologyOnK.h"
#include "LocalSystemSL2Z.h"
#include "LocalSystemSimplicialComplex.h"
#include "Hecke/HeckeCosetTables.h"
#include "Hecke/HeckeOperatorOnK.h"
#include "Hecke/HeckeTauTable.h"
#include "Hecke/SL2ZTessellationTables.h"
#include "../Cuda/CudaCtx.h"
#include "../Engine/EngineRuntime.h"
#include "../Engine/EngineTestUtil.h"
#include "../Engine/Executor.h"
#include "../Engine/Recorder.h"
#include "../Engine/ThreadPool.h"
#include "../ModArithmetic/DenseBuf.h"
#include "../ModArithmetic/ModCtx.h"
#include "../ModArithmetic/RetiredDeviceLevers.h"   // (S25 LEVERS-25) gpu_lean_lever()

#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace {

using namespace engine;
using namespace engine::tape;
using namespace engine::tape::test;
using kernels::GemmOp;
namespace cu = engine::cuda;

uint32_t g_checks = 0;
uint32_t g_failures = 0;

void Check(bool ok, const std::string& what) {
    ++g_checks;
    if (!ok) {
        ++g_failures;
        if (g_failures <= 40) std::cout << "  [FAIL] " << what << std::endl;
    }
}

// (S25 LEVERS-25) the arms switch ENGINE_GPU_LEAN by its Scoped override (the env arm is retired in OFF builds).

enum class Arm { Cpu, GpuLean, GpuNoLean };
const char* arm_name(Arm a) { return a == Arm::Cpu ? "cpu" : a == Arm::GpuLean ? "gpu lean" : "gpu lean=0"; }

PassOptions opts_for(Arm a) {
    PassOptions o;
    if (a != Arm::Cpu) { o.device_policy = Device::GPU; o.device_model = true; o.policy = DevicePolicy::Gpu; }
    return o;
}

uint64_t hash_slots(uint64_t h, const RunResult& rr) {
    for (const ResultSlot& s : rr.slots) {
        h = fnv_mix(h, s.written ? 1 : 0); h = fnv_mix(h, s.rank); h = fnv_mix(h, s.equal ? 1 : 0);
        for (uint32_t p : s.pivots) h = fnv_mix(h, p);
    }
    return h;
}

// ---- L / S: the engine-level programs ------------------------------------------------------------------------------
struct ProgOut { uint64_t hash = 0; uint64_t lean_runs = 0; uint64_t dev_nodes = 0; uint32_t gpu_compute = 0, gpu_nodes = 0;
                 size_t live_delta = 0; uint64_t allocs_delta = 0; };

// The memory-only program: T = 0; T += a X; T += b I; T *= c; U = T^T; compare(T, Y). Outputs U; one result slot.
// The compute program: C = X Y; C += a X; compare(C, X). Outputs C; one result slot.
template<class Elt>
ProgOut run_prog(Executor<Elt>& ex, const ModCtx<Elt>& ctx, bool compute, Arm arm, const DenseBuf<Elt>& X, const DenseBuf<Elt>& Y,
                 Elt a, Elt b, Elt c, bool session) {
    const uint32_t d = X.rows;
    DenseBuf<Elt> U(d, d);
    Recorder<Elt> rec(ctx);
    const BufId bX = rec.input(cview(X)), bY = rec.input(cview(Y));
    const BufId bU = rec.output(view(U));
    if (!compute) {
        const BufId t = rec.temp(d, d);
        rec.zero(rec.whole(t));
        rec.axpy(a, rec.whole(bX), rec.whole(t));
        rec.axpy_identity(b, rec.whole(t));
        rec.scal(c, rec.whole(t));
        rec.transpose(rec.whole(t), rec.whole(bU));
        rec.compare(rec.whole(t), rec.whole(bY));
    } else {
        rec.gemm(rec.whole(bX), rec.whole(bY), rec.whole(bU), GemmOp::Set);
        rec.axpy(a, rec.whole(bX), rec.whole(bU));
        rec.compare(rec.whole(bU), rec.whole(bX));
    }
    Program<Elt> prog = rec.finish();
    const uint64_t lean0 = ex.stats().lean_host_runs, dev0 = ex.device_stats().gpu_nodes;
    cu::ArenaStats as0{};
    const bool dev = cu::available();
    if (dev) as0 = cu::CudaCtx::get().arena_stats();
    if (session) { Check(ex.elim_session_begin(), "S: elim_session_begin opened a session"); }
    const RunResult rr = ex.run(prog, rec.bindings(), opts_for(arm));
    ProgOut o;
    if (dev) {
        const cu::ArenaStats as1 = cu::CudaCtx::get().arena_stats();
        o.live_delta = as1.live_bytes > as0.live_bytes ? as1.live_bytes - as0.live_bytes : 0;
        o.allocs_delta = (as1.cuda_mallocs + as1.reuses) - (as0.cuda_mallocs + as0.reuses);
    }
    if (session) ex.elim_session_end(false);
    o.hash = hash_slots(hash_view(cview(U)), rr);
    o.lean_runs = ex.stats().lean_host_runs - lean0;
    o.dev_nodes = ex.device_stats().gpu_nodes - dev0;
    o.gpu_compute = prog.stats.gpu_compute_nodes;
    o.gpu_nodes = prog.stats.gpu_nodes;
    return o;
}

template<class Elt>
void arm_L(Elt p, uint64_t seed, const std::string& tag) {
    const ModCtx<Elt> ctx(p);
    Rng rng(seed);
    const uint32_t d = 192;
    const DenseBuf<Elt> X = random_buf<Elt>(rng, d, d, p), Y = random_buf<Elt>(rng, d, d, p);
    const Elt a = rng.nonzero<Elt>(p), b = rng.nonzero<Elt>(p), c = rng.nonzero<Elt>(p);
    ThreadPool pool(4);
    for (int compute = 0; compute < 2; ++compute) {
        const std::string what = tag + (compute ? " compute program" : " memory-only program");
        uint64_t ref = 0;
        {
            Executor<Elt> ex(&pool);
            const ProgOut o = run_prog<Elt>(ex, ctx, compute != 0, Arm::Cpu, X, Y, a, b, c, false);
            ref = o.hash;
            Check(o.gpu_nodes == 0 && o.gpu_compute == 0 && o.lean_runs == 0, "L " + what + ": cpu compiles no GPU node, no lean run");
        }
        for (Arm arm : {Arm::GpuLean, Arm::GpuNoLean}) {
            for (int verify = 0; verify < 2; ++verify) {
                engine::lever::EnvLever::Scoped ln(engine::lever::gpu_lean_lever(), arm == Arm::GpuLean);   // (S25 LEVERS-25) was the env arm
                Executor<Elt> ex(&pool);
                ex.set_verify_device(verify != 0);
                const std::string w = "L " + what + " (" + arm_name(arm) + (verify ? ", verify" : "") + ")";
                const ProgOut o = run_prog<Elt>(ex, ctx, compute != 0, arm, X, Y, a, b, c, false);
                Check(o.hash == ref, w + ": outputs and result slots == cpu");
                Check(o.gpu_nodes > 0, w + ": compiled with GPU nodes");
                if (!compute) {
                    Check(o.gpu_compute == 0, w + ": no GPU compute node");
                    if (arm == Arm::GpuLean) {
                        Check(o.lean_runs == 1, w + ": one lean host run");
                        Check(o.dev_nodes == 0, w + ": no node executed on the device");
                        Check(o.live_delta == 0 && o.allocs_delta == 0, w + ": no device byte allocated");
                    } else {
                        Check(o.lean_runs == 0, w + ": no lean host run under lean=0");
                        Check(o.dev_nodes > 0, w + ": nodes executed on the device under lean=0 (the contrast)");
                    }
                } else {
                    Check(o.gpu_compute > 0, w + ": GPU compute nodes compiled");
                    Check(o.lean_runs == 0, w + ": a compute program is never lean");
                    Check(o.dev_nodes > 0, w + ": nodes executed on the device");
                }
            }
        }
        if (!compute) {   // S: inside an elimination session the rule is off
            engine::lever::EnvLever::Scoped ln(engine::lever::gpu_lean_lever(), true);   // (S25 LEVERS-25) was the env arm
            Executor<Elt> ex(&pool);
            const ProgOut o = run_prog<Elt>(ex, ctx, false, Arm::GpuLean, X, Y, a, b, c, true);
            Check(o.hash == ref, "S " + what + ": outputs == cpu inside a session");
            Check(o.lean_runs == 0, "S " + what + ": no lean host run inside a session");
            Check(o.dev_nodes > 0, "S " + what + ": the session program ran on the device");
        }
    }
}

// ---- H: the Hecke operator ------------------------------------------------------------------------------------------
uint64_t hash_flat(uint64_t h, const FlatMatrix& F) {
    h = fnv_mix(h, F.get_rows()); h = fnv_mix(h, F.get_cols());
    for (uint32_t i = 0; i < F.get_rows(); ++i)
        for (uint32_t j = 0; j < F.get_cols(); ++j) h = fnv_mix(h, F.get_entry(i, j));
    return h;
}

uint64_t hecke_digest(const SimplicialComplex& K, const hecke::SL2ZTessellationTables& model, uint32_t ell, uint32_t n_sym, uint32_t p) {
    std::vector<LocalSystemSimplicialComplex::SimplexRelation> rels;
    for (const auto& datum : GenerateSL2ZRelationsWithGroupData(K, n_sym, ell)) rels.push_back(datum.rel);
    LocalSystemSimplicialComplex lssc(K, rels, n_sym + 1, ell);
    EquivariantCohomologyOnK coh(&lssc);
    coh.compute_cohomology();
    const hecke::HeckeCosetTables cosets(p, 1, ell, n_sym);
    const hecke::HeckeTauTable tau(K, model, cosets, ell, n_sym);
    hecke::HeckeOperatorOnK op(&coh, &cosets, &tau);
    op.compute();
    uint64_t h = fnv_init();
    for (const auto& F : op.get_Theta_flat_matrices()) h = hash_flat(h, F);
    for (const auto& F : op.get_Theta_eq_matrices()) h = hash_flat(h, F);
    for (const auto& A : op.get_H_matrices()) {
        h = fnv_mix(h, A.rows); h = fnv_mix(h, A.cols);
        for (uint32_t i = 0; i < A.rows; ++i)
            for (uint32_t j = 0; j < A.cols; ++j) h = fnv_mix(h, A.data[i][j]);
    }
    return h;
}

void arm_H(uint64_t& hecke_lean_runs) {
    const SimplicialComplex K = CreateSL2ZFundamentalDomainTriangulation();
    const hecke::SL2ZTessellationTables model(K);
    struct Cell { uint32_t ell, n, p; };
    for (const Cell& c : {Cell{13, 24, 7}, Cell{7, 30, 5}}) {
        std::ostringstream os; os << "H (ell, n, p) = (" << c.ell << ", " << c.n << ", " << c.p << ")";
        const std::string tag = os.str();
        uint64_t ref = 0;
        {
            ScopedThreadDevice cpu(ThreadDevice::Cpu);
            const uint64_t l0 = gpu_lean_host_runs();
            ref = hecke_digest(K, model, c.ell, c.n, c.p);
            Check(gpu_lean_host_runs() == l0, tag + " cpu: no lean host run");
        }
        for (Arm arm : {Arm::GpuLean, Arm::GpuNoLean}) {
            engine::lever::EnvLever::Scoped ln(engine::lever::gpu_lean_lever(), arm == Arm::GpuLean);   // (S25 LEVERS-25) was the env arm
            DeviceTally tally;
            ScopedThreadDevice gpu(ThreadDevice::Gpu, &tally);
            const uint64_t l0 = gpu_lean_host_runs();
            const uint64_t h = hecke_digest(K, model, c.ell, c.n, c.p);
            const uint64_t lr = gpu_lean_host_runs() - l0;
            Check(h == ref, tag + " " + arm_name(arm) + ": Theta_flat, Theta_eq, A == cpu");
            const uint64_t gp = tally.gpu_programs.load(), lp = tally.lean_host_programs.load();
            Check(gp > 0, tag + " " + arm_name(arm) + ": programs compiled with GPU nodes");
            // (audit R4) the tally separates compiled placement (gpu_programs) from host execution (lean_host_programs)
            Check(lp == lr && lp <= gp, tag + " " + arm_name(arm) + ": the tally's lean_host_programs == the lean host runs, <= gpu_programs");
            Check(gp > lp, tag + " " + arm_name(arm) + ": programs executed on the device remain (gpu_programs - lean_host_programs > 0)");
            if (arm == Arm::GpuLean) { Check(lr > 0, tag + " gpu lean: route 2's memory-only programs ran on the host"); hecke_lean_runs += lr; }
            else Check(lr == 0, tag + " gpu lean=0: no lean host run");
            std::cout << "  " << tag << " " << std::left << std::setw(11) << arm_name(arm) << ": digest "
                      << (h == ref ? "== cpu" : "DIFFERS") << ", gpu-compiled programs " << gp << ", of which lean host runs " << lp
                      << ", device runs " << (gp - lp) << std::endl;
        }
    }
}

// ---- A: the arena slack -----------------------------------------------------------------------------------------------
void arm_A(size_t& slack_mb) {
    cu::CudaCtx& cx = cu::CudaCtx::get();
    const size_t slack = cx.arena_idle_slack();
    slack_mb = slack >> 20;
    // the card's total through the public device info (cudaDeviceProp; the context used cudaMemGetInfo's total, which
    // may differ by a few MB: hence the 3 % tolerance below)
    const size_t tot = cu::device_info().total_mem;
    auto near = [&](size_t target) { return tot != 0 && slack * 100 >= target * 97 && slack * 100 <= target * 103; };
    const char* env = std::getenv("ENGINE_CUDA_ARENA_SLACK_MB");
    if (env == nullptr || *env == '\0') {
        // read once at context creation: a sixteenth under the lean default, a quarter under ENGINE_GPU_LEAN=0
        const bool sixteenth = near(tot / 16), quarter = near(tot / 4);
        Check(sixteenth || quarter, "A: the default slack is a sixteenth or a quarter of the card");
        const char* ln = std::getenv("ENGINE_GPU_LEAN");
        if (ln == nullptr || ln[0] != '0') Check(sixteenth || tot == 0, "A: the slack is a sixteenth of the card under the lean default");
    }
    // churn: 72 distinct block sizes 8 .. 79 MB, allocated and freed one after the other (the census pattern: every
    // operator / cone shape its own size); idle may exceed the post-trim idle by at most the slack plus two blocks.
    cx.release_cached(false);
    const cu::ArenaStats s0 = cx.arena_stats();
    const size_t idle0 = s0.reserved_bytes - std::min(s0.reserved_bytes, s0.live_bytes);
    size_t worst = 0;
    const size_t maxblk = size_t(79) << 20;
    for (uint32_t i = 0; i < 72; ++i) {
        const size_t bytes = (size_t(8 + i) << 20) + 4096;
        void* q = cx.dalloc(bytes);
        cx.dfree(q, bytes);
        const cu::ArenaStats s = cx.arena_stats();
        worst = std::max(worst, s.reserved_bytes - std::min(s.reserved_bytes, s.live_bytes));
    }
    Check(worst <= idle0 + slack + 2 * maxblk, "A: idle arena bytes stay within the slack plus two blocks over the churn");
    cx.release_cached(false);
    std::cout << "  A slack " << slack_mb << " MB (card " << (tot >> 20) << " MB), churn idle peak " << (worst >> 20)
              << " MB over a post-trim idle of " << (idle0 >> 20) << " MB" << std::endl;
}

} // namespace

uint32_t GpuLeanSessionValidation() {
    g_checks = 0; g_failures = 0;
    if (!cu::available()) {
        std::cout << "[SKIP] GpuLeanSessionValidation: no usable CUDA device (" << cu::unavailable_reason()
                  << "); the VRAM-lean rule is reached only under ENGINE_DEVICE=gpu" << std::endl;
        return 0;
    }
    const auto t0 = std::chrono::steady_clock::now();
    uint64_t hecke_lean_runs = 0;
    size_t slack_mb = 0;
    try {
        arm_L<uint32_t>(13u, 0x5037A11ull, "[u32 p=13]");
        arm_L<uint32_t>(134217757u, 0x5037A12ull, "[u32 p=134217757]");
        arm_L<uint64_t>(134217757ull, 0x5037A13ull, "[u64 p=134217757]");
        arm_H(hecke_lean_runs);
        arm_A(slack_mb);
    } catch (const std::exception& e) {
        Check(false, std::string("exception: ") + e.what());
    }
    const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    std::cout << "GPU_LEAN checks=" << g_checks << " failing=" << g_failures << " hecke_lean_runs=" << hecke_lean_runs
              << " slack_mb=" << slack_mb << " wall_s=" << std::fixed << std::setprecision(2) << s << std::endl;
    return g_failures;
}
