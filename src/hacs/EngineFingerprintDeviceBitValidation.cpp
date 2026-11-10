// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// EngineFingerprintDeviceBitValidation.cpp -- see the header.
//
// hecke-extensions slot 10, WP-8: decision 10 (A), the ninth fingerprint bit.
#include "EngineFingerprintDeviceBitValidation.h"

#include <iostream>
#include <string>
#include <utility>

#include "../Engine/EngineRuntime.h"
#include "../Engine/Executor.h"
#include "../Engine/ProgramCache.h"
#include "../Engine/Recorder.h"
#include "../ModArithmetic/DenseBuf.h"
#include "../ModArithmetic/DenseView.h"
#include "../ModArithmetic/EngineConfig.h"
#include "../ModArithmetic/ModCtx.h"

namespace {

using namespace engine;
using namespace engine::tape;

struct Checker {
    uint32_t checks = 0, fails = 0;
    void check(bool ok, const std::string& what) {
        ++checks;
        if (!ok) {
            ++fails;
            std::cout << "  FAIL: " << what << "\n";
        }
    }
};

// The three from_env() device mappings (RunMode / EngineElimPassOptionsValidation's own),
// everything else at from_env()'s default.
PassOptions MappingOf(DevicePolicy pol) {
    PassOptions o = PassOptions::from_env();
    if (pol == DevicePolicy::Cpu) {
        o.device_policy = Device::CPU;
        o.device_model = false;
        o.policy = DevicePolicy::Cpu;
    } else if (pol == DevicePolicy::Gpu) {
        o.device_policy = Device::GPU;
        o.device_model = true;
        o.policy = DevicePolicy::Gpu;
    } else {
        o.device_policy = Device::GPU;
        o.device_model = true;
        o.policy = DevicePolicy::Hybrid;
    }
    return o;
}

constexpr uint32_t kP = 1000003u;

// A tiny one-GEMM program, recorded IDENTICALLY on every call (same shape,
// same entries), compiled and run under `pol` via run_cached().  The three
// policies share one base ShapeKey (tag "wp8_fingerprint_probe", extra = 0),
// so the only way their compiled programs can land in distinct cache slots is
// run_cached()'s own pass_options_fingerprint() fold -- exactly what this
// validation is checking.
CachedRun RunOnce(DevicePolicy pol) {
    using Elt = uint32_t;
    const ModCtx<Elt> ctx(kP);
    const uint32_t m = 6;
    DenseBuf<Elt> A(m, m), B(m, m), C(m, m);
    for (uint32_t i = 0; i < m; ++i)
        for (uint32_t j = 0; j < m; ++j) {
            A.at(i, j) = (i * 7 + j + 1) % kP;
            B.at(i, j) = (i + 3 * j + 2) % kP;
        }
    Recorder<Elt> rec(ctx);
    const BufId a = rec.input(cview(A));
    const BufId b = rec.input(cview(B));
    const BufId o = rec.output(view(C));
    rec.gemm(rec.whole(a), rec.whole(b), rec.whole(o), engine::kernels::GemmOp::Set);
    Program<Elt> prog = rec.finish();
    Executor<Elt> ex(&consumer_pool());
    ShapeKey key = make_key<Elt>("wp8_fingerprint_probe", ctx, m, 0);
    key.row_dims = {m};
    key.col_dims = {m};
    ScopedPassOptions so(MappingOf(pol));
    return run_cached(ex, key, std::move(prog), rec.bindings());
}

} // namespace

uint32_t EngineFingerprintDeviceBitValidation() {
    std::cout << "=============== EngineFingerprintDeviceBitValidation ===============\n";
    Checker g;

    // ---- (A) fingerprint distinctness, and the exact bit pattern -------------
    const uint64_t fcpu = pass_options_fingerprint(MappingOf(DevicePolicy::Cpu));
    const uint64_t fhyb = pass_options_fingerprint(MappingOf(DevicePolicy::Hybrid));
    const uint64_t fgpu = pass_options_fingerprint(MappingOf(DevicePolicy::Gpu));
    std::cout << std::hex << "  (A) fingerprints: cpu=0x" << fcpu << " hybrid=0x" << fhyb << " gpu=0x" << fgpu
              << std::dec << "\n";
    g.check(fcpu != fhyb && fhyb != fgpu && fcpu != fgpu, "(A) cpu / hybrid / gpu give three distinct fingerprints");
    g.check((fcpu & 384u) == 0u, "(A) cpu: neither device bit (128, 256) set");
    g.check((fhyb & 384u) == 128u, "(A) hybrid: bit 128 set, bit 256 clear");
    g.check((fgpu & 384u) == 384u, "(A) gpu: bits 128 and 256 both set");
    g.check((fcpu & 127u) == (fhyb & 127u) && (fhyb & 127u) == (fgpu & 127u),
            "(A) the seven non-device bits agree across cpu / hybrid / gpu at the same PassOptions otherwise");

    // ---- (B) three compiled programs, three distinct cache slots -------------
    // Executor::run() falls back to the CPU for a Device::GPU-assigned node
    // when device_usable() is false, so this needs no CUDA device and never
    // [SKIP]s (src/Engine/Executor.cpp: the CPU loop's
    // `if (gpu && n.device == Device::GPU) continue;` only skips a GPU node
    // when a real device backend answered `begin_run()`).
    if (engine::backend_is_legacy()) {
        std::cout << "  (B) [SKIP] ENGINE_BACKEND=legacy: no program is recorded\n";
    } else {
        const std::size_t before = ProgramCache<uint32_t>::global().size();
        const CachedRun cpu1 = RunOnce(DevicePolicy::Cpu);
        const CachedRun hyb1 = RunOnce(DevicePolicy::Hybrid);
        const CachedRun gpu1 = RunOnce(DevicePolicy::Gpu);
        g.check(cpu1.inserted && hyb1.inserted && gpu1.inserted,
                "(B) the first compile under each policy is a fresh insert (no cross-policy hit)");
        const CachedRun cpu2 = RunOnce(DevicePolicy::Cpu);
        const CachedRun hyb2 = RunOnce(DevicePolicy::Hybrid);
        const CachedRun gpu2 = RunOnce(DevicePolicy::Gpu);
        g.check(cpu2.hit && hyb2.hit && gpu2.hit,
                "(B) the second compile under the SAME policy replays its own slot (a hit)");
        const std::size_t after = ProgramCache<uint32_t>::global().size();
        std::cout << "  (B) program cache size " << before << " -> " << after << " (+3 expected: one slot per policy)\n";
        g.check(after == before + 3, "(B) exactly three new cache slots, one per policy");
    }

    std::cout << "EngineFingerprintDeviceBitValidation: " << g.checks << " checks, " << g.fails << " failing\n";
    return g.fails;
}
