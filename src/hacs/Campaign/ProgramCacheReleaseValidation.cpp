// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// Campaign/ProgramCacheReleaseValidation.cpp -- see the header (slot-8 WP-5).
#include "ProgramCacheReleaseValidation.h"

#include "ProgramCacheRelease.h"

#include <atomic>
#include <chrono>
#include <iostream>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include "../../Engine/EngineRuntime.h"
#include "../../Engine/Executor.h"
#include "../../Engine/ProgramCache.h"
#include "../../Engine/Recorder.h"
#include "../../ModArithmetic/DenseBuf.h"
#include "../../ModArithmetic/EngineConfig.h"
#include "../../ModArithmetic/ModCtx.h"
#include "../EquivariantCohomologyOnK.h"
#include "../Hecke/HeckeCosetTables.h"
#include "../Hecke/HeckeOperatorOnK.h"
#include "../Hecke/HeckeTauTable.h"
#include "../Hecke/SL2ZTessellationTables.h"
#include "../LocalSystemSL2Z.h"
#include "../LocalSystemSimplicialComplex.h"
#include "../SimplicialCohomology.h"

namespace {

using engine::tape::ProgramCache;

struct Checker {
    uint64_t checks = 0, fails = 0;
    void check(bool ok, const std::string& what) {
        ++checks;
        if (!ok) { ++fails; std::cout << "  FAIL: " << what << std::endl; }
    }
};

// ---- a pipeline point: cohomology + A^{(q)}(T_p), as a comparable text ------
// Every point owns its complex and tessellation tables (the reason is in
// EngineTapeConsumerValidation.cpp: SimplicialComplex caches Z1 for the first
// prime it is asked for).
std::string PipelinePoint(uint32_t n, uint32_t ell, uint64_t p) {
    SimplicialComplex K = CreateSL2ZFundamentalDomainTriangulation();
    const hecke::SL2ZTessellationTables model(K);
    std::vector<LocalSystemSimplicialComplex::SimplexRelation> rels;
    for (const auto& datum : GenerateSL2ZRelationsWithGroupData(K, n, ell)) rels.push_back(datum.rel);
    LocalSystemSimplicialComplex lssc(K, rels, n + 1, ell);
    EquivariantCohomologyOnK coh(&lssc);
    coh.compute_cohomology();
    hecke::HeckeCosetTables cosets(p, 1, ell, n);
    hecke::HeckeTauTable tau(K, model, cosets, ell, n);
    hecke::HeckeOperatorOnK op(&coh, &cosets, &tau);
    op.compute();
    std::string out = "dims";
    for (uint32_t d : coh.get_H_eq_dimensions()) out += " " + std::to_string(d);
    out += "\n";
    out += op.to_json();
    return out;
}

// ---- a uint64_t tape program through run_cached (the uint64_t cache) --------
constexpr uint64_t kP64 = 4294967311ull;   // 2^32 + 15, prime: the uint64_t regime
std::vector<uint64_t> TapeGemm64() {
    using Elt = uint64_t;
    const engine::ModCtx<Elt> ctx(kP64);
    const uint32_t m = 12;
    engine::DenseBuf<Elt> A(m, m), B(m, m), C(m, m);
    uint64_t s = 0x9e3779b97f4a7c15ull;
    for (uint32_t i = 0; i < m; ++i)
        for (uint32_t j = 0; j < m; ++j) {
            s = s * 6364136223846793005ull + 1442695040888963407ull;
            A.at(i, j) = (s >> 11) % kP64;
            s = s * 6364136223846793005ull + 1442695040888963407ull;
            B.at(i, j) = (s >> 11) % kP64;
        }
    engine::tape::Recorder<Elt> rec(ctx);
    const engine::tape::BufId a = rec.input(engine::cview(A));
    const engine::tape::BufId b = rec.input(engine::cview(B));
    const engine::tape::BufId o = rec.output(engine::view(C));
    rec.gemm(rec.whole(a), rec.whole(b), rec.whole(o), engine::kernels::GemmOp::Set);
    engine::tape::Program<Elt> prog = rec.finish();
    engine::tape::Executor<Elt> ex(&engine::tape::consumer_pool());
    engine::tape::ShapeKey key = engine::tape::make_key<Elt>("wp5_release_probe", ctx, m, 0);
    key.row_dims = {m};
    key.col_dims = {m};
    engine::tape::run_cached(ex, key, std::move(prog), rec.bindings());
    std::vector<uint64_t> out;
    for (uint32_t i = 0; i < m; ++i)
        for (uint32_t j = 0; j < m; ++j) out.push_back(C.at(i, j));
    return out;
}

size_t Size32() { return ProgramCache<uint32_t>::global().size(); }
size_t Size64() { return ProgramCache<uint64_t>::global().size(); }

} // namespace

uint32_t ProgramCacheReleaseValidation() {
    std::cout << "================= ProgramCacheReleaseValidation =================\n";
    if (engine::backend_is_legacy()) {
        std::cout << "ProgramCacheReleaseValidation: SKIP (ENGINE_BACKEND=legacy: no program is recorded)\n";
        return 0;
    }
    const auto t0 = std::chrono::steady_clock::now();
    Checker g;
    const uint32_t kN = 16, kEll = 691;
    const uint64_t kP = 2;

    // ---- (A) the policy ----------------------------------------------------
    campaign::ReleaseEngineProgramCaches();
    g.check(Size32() == 0 && Size64() == 0 && campaign::EngineProgramCacheSize() == 0,
            "(A) both caches empty after a release");
    g.check(campaign::ReleaseEngineProgramCaches() == 0, "(A) releasing empty caches drops nothing");

    const std::string ref_point = PipelinePoint(kN, kEll, kP);
    const std::vector<uint64_t> ref_gemm = TapeGemm64();
    const size_t n32 = Size32(), n64 = Size64();
    std::cout << "  cached after one point: uint32_t " << n32 << ", uint64_t " << n64 << "\n";
    g.check(n32 > 0, "(A) the pipeline point populated ProgramCache<uint32_t>");
    g.check(n64 == 1, "(A) the uint64_t tape program populated ProgramCache<uint64_t>");
    g.check(campaign::EngineProgramCacheSize() == n32 + n64, "(A) EngineProgramCacheSize is the sum");
    g.check(campaign::ReleaseEngineProgramCaches(n32 + n64) == 0 && Size32() == n32 && Size64() == n64,
            "(A) size == threshold: nothing released");
    g.check(campaign::ReleaseEngineProgramCaches(n32 + n64 - 1) == n32 + n64 && Size32() == 0 &&
                Size64() == 0,
            "(A) size > threshold: both caches cleared, the count returned");

    // ---- (B) a recompile after a clear moves no value -----------------------
    {
        const auto s32 = ProgramCache<uint32_t>::global().stats();
        const auto s64 = ProgramCache<uint64_t>::global().stats();
        const std::string again = PipelinePoint(kN, kEll, kP);
        const std::vector<uint64_t> again64 = TapeGemm64();
        g.check(ProgramCache<uint32_t>::global().stats().misses > s32.misses,
                "(B) the uint32_t point recompiled (missed) after the clear");
        g.check(ProgramCache<uint64_t>::global().stats().misses > s64.misses,
                "(B) the uint64_t program recompiled (missed) after the clear");
        g.check(again == ref_point, "(B) pipeline point after a clear == first run (dims + hecke JSON)");
        g.check(again64 == ref_gemm, "(B) uint64_t tape program after a clear == first run");
        const auto h32 = ProgramCache<uint32_t>::global().stats();
        const std::string replay = PipelinePoint(kN, kEll, kP);
        g.check(ProgramCache<uint32_t>::global().stats().hits > h32.hits,
                "(B) the next run replays (hits) again");
        g.check(replay == ref_point, "(B) replay == first run");
    }

    // ---- (C) clear-while-running: two threads ---------------------------------
    {
        const int kRepeats = 6;
        std::atomic<bool> done{false};
        std::atomic<int> worker_fail{0};
        std::string worker_error;
        std::thread worker([&] {
            try {
                for (int r = 0; r < kRepeats; ++r) {
                    if (PipelinePoint(kN, kEll, kP) != ref_point) worker_fail.fetch_add(1);
                    if (TapeGemm64() != ref_gemm) worker_fail.fetch_add(1);
                }
            } catch (const std::exception& e) {
                worker_error = e.what();
                worker_fail.fetch_add(1000);
            }
            done.store(true, std::memory_order_release);
        });
        uint64_t calls = 0, effective = 0, released = 0;
        while (!done.load(std::memory_order_acquire)) {
            const size_t k = campaign::ReleaseEngineProgramCaches();
            ++calls;
            if (k > 0) { ++effective; released += k; }
            std::this_thread::yield();
        }
        worker.join();
        std::cout << "  clear-while-running: " << kRepeats << " repeats, " << calls << " release calls, "
                  << effective << " effective, " << released << " programs dropped mid-run\n";
        if (!worker_error.empty()) std::cout << "  worker exception: " << worker_error << "\n";
        g.check(worker_fail.load() == 0, "(C) every repeat under concurrent clears == the reference");
        g.check(effective > 0, "(C) at least one clear released programs while the worker ran");
    }
    campaign::ReleaseEngineProgramCaches();

    const double secs =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    std::cout << "ProgramCacheReleaseValidation: " << g.checks << " checks, " << g.fails
              << " failing (" << secs << " s)\n";
    return g.fails == 0 ? 0u : 1u;
}
