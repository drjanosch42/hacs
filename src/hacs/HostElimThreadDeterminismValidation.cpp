// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// hecke-extensions slot 23, PERF-P2 -- HostElimThreadDeterminismValidation (see the header).

#include "HostElimThreadDeterminismValidation.h"

#include "EquivariantCohomologyOnK.h"
#include "EquivariantMappingCone.h"
#include "LocalSystemSL2Z.h"
#include "LocalSystemSimplicialComplex.h"
#include "Hecke/HeckeInvolution.h"
#include "Hecke/SL2ZBoundaryPair.h"
#include "Hecke/SL2ZTessellationTables.h"
#include "../Engine/EngineRuntime.h"
#include "../Engine/ThreadPool.h"
#include "../ModArithmetic/DenseBuf.h"
#include "../ModArithmetic/DenseElim.h"
#include "../ModArithmetic/DenseView.h"
#include "../ModArithmetic/FlatMatrix.h"
#include "../ModArithmetic/Fnv.h"
#include "../ModArithmetic/ModCtx.h"

#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <functional>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using engine::DenseBuf;
using engine::ModCtx;
using engine::tape::ScopedConsumerPool;
using engine::tape::ScopedHostElimHook;
using engine::tape::ThreadPool;

uint32_t g_checks = 0;
uint32_t g_failures = 0;

void Check(bool ok, const std::string& what) {
    ++g_checks;
    if (!ok) {
        ++g_failures;
        if (g_failures <= 40) std::cout << "  [FAIL] " << what << std::endl;
    }
}

std::string hex(uint64_t v) {
    std::ostringstream os;
    os << "0x" << std::hex << std::setw(16) << std::setfill('0') << v;
    return os.str();
}

uint64_t hash_dense(uint64_t h, const DenseBuf<uint32_t>& d) {
    h = engine::FnvU64(h, d.rows);
    h = engine::FnvU64(h, d.cols);
    for (uint32_t i = 0; i < d.rows; ++i)
        for (uint32_t j = 0; j < d.cols; ++j) h = engine::FnvU64(h, d.at(i, j));
    return h;
}

uint64_t hash_flat(uint64_t h, const FlatMatrix& F) {
    h = engine::FnvU64(h, F.get_rows());
    h = engine::FnvU64(h, F.get_cols());
    for (uint32_t i = 0; i < F.get_rows(); ++i)
        for (uint32_t j = 0; j < F.get_cols(); ++j) h = engine::FnvU64(h, F.get_entry(i, j));
    return h;
}

uint64_t hash_pivots(uint64_t h, const engine::elim::RrefResult& r) {
    h = engine::FnvU64(h, r.rank);
    for (uint32_t c : r.pivots) h = engine::FnvU64(h, c);
    return h;
}

// A rank-deficient m x n matrix mod p (deterministic): random columns, then every fourth column a combination of two
// earlier ones, and the last eighth of the rows a combination of two earlier rows -- so the PLU meets both column and
// row dependencies, and every pivot bookkeeping branch runs.
DenseBuf<uint32_t> test_matrix(uint32_t m, uint32_t n, uint32_t p, uint64_t seed) {
    DenseBuf<uint32_t> A(m, n);
    uint64_t s = seed;
    auto next = [&]() { s = s * 6364136223846793005ull + 1442695040888963407ull; return static_cast<uint32_t>(s >> 33); };
    for (uint32_t i = 0; i < m; ++i)
        for (uint32_t j = 0; j < n; ++j) A.at(i, j) = next() % p;
    for (uint32_t j = 3; j < n; j += 4) {
        const uint32_t a = next() % j, b = next() % j, ca = next() % p, cb = next() % p;
        for (uint32_t i = 0; i < m; ++i)
            A.at(i, j) = static_cast<uint32_t>((static_cast<uint64_t>(ca) * A.at(i, a) + static_cast<uint64_t>(cb) * A.at(i, b)) % p);
    }
    for (uint32_t i = m - m / 8; i < m; ++i) {
        const uint32_t a = next() % (m - m / 8), b = next() % (m - m / 8), ca = next() % p, cb = next() % p;
        for (uint32_t j = 0; j < n; ++j)
            A.at(i, j) = static_cast<uint32_t>((static_cast<uint64_t>(ca) * A.at(a, j) + static_cast<uint64_t>(cb) * A.at(b, j)) % p);
    }
    return A;
}

// The direct arm on the CALLING thread: rref, column_rank_profile, rref_with_transform of one matrix.  Returns one digest
// over every output entry and pivot list.
uint64_t direct_arm(const DenseBuf<uint32_t>& A0, uint32_t p) {
    const ModCtx<uint32_t> ctx(p);
    uint64_t h = engine::kFnvOffset64;
    {
        DenseBuf<uint32_t> R = A0;
        const engine::elim::RrefResult r = engine::elim::rref<uint32_t>(ctx, engine::view(R));
        h = hash_pivots(h, r);
        h = hash_dense(h, R);
    }
    {
        DenseBuf<uint32_t> W = A0;
        h = hash_pivots(h, engine::elim::column_rank_profile<uint32_t>(ctx, engine::view(W)));
    }
    {
        DenseBuf<uint32_t> R(A0.rows, A0.cols), U(A0.rows, A0.rows);
        const engine::elim::RrefResult r =
            engine::elim::rref_with_transform<uint32_t>(ctx, engine::cview(A0), engine::view(R), engine::view(U));
        h = hash_pivots(h, r);
        h = hash_dense(h, R);
        h = hash_dense(h, U);
    }
    return h;
}

struct PipelineDigests {
    uint64_t cone = 0;   // golden_digest() + Z / B blocks
    uint64_t iota = 0;   // every +- basis
    uint32_t dimHc1 = 0, dimHplus1 = 0, dimHminus1 = 0;
    uint64_t regions_iota = 0, regions_cone = 0;   // hook regions inside iota.compute() / make_cone() (sequential callers only)
};

uint64_t hash_block(uint64_t h, const engine::BlockMat<uint32_t>& M) {
    return hash_dense(h, M.to_dense(engine::Layout::RowMajor));
}

// One (n, ell) cell, every object built from scratch on the current consumer pool.
PipelineDigests pipeline_cell(const SimplicialComplex& K, const hecke::SL2ZTessellationTables& model, uint32_t n, uint32_t ell) {
    PipelineDigests out;
    const std::vector<SL2ZRelationDatum> data = GenerateSL2ZRelationsWithGroupData(K, n, ell);
    std::vector<LocalSystemSimplicialComplex::SimplexRelation> rels;
    rels.reserve(data.size());
    for (const auto& d : data) rels.push_back(d.rel);
    LocalSystemSimplicialComplex lssc(K, rels, n + 1, ell);
    EquivariantCohomologyOnK coh(&lssc);
    coh.compute_cohomology();

    // The involution iota^* (bulk), as the campaign's sign split builds it.
    hecke::HeckeInvolution iota(&coh, hecke::BuildIotaSimplicialMapOnModel(K, model), "iota^* (bulk)");
    iota.set_exact_relation_data(data);
    const uint64_t ri0 = engine::tape::host_elim_hook_regions();
    iota.compute();
    out.regions_iota = engine::tape::host_elim_hook_regions() - ri0;
    uint64_t hi = engine::kFnvOffset64;
    for (uint32_t q = 0; q <= iota.q_max(); ++q) {
        hi = hash_flat(hi, iota.C_eq_plus_basis(q));
        hi = hash_flat(hi, iota.C_eq_minus_basis(q));
        hi = hash_flat(hi, iota.Z_eq_plus_basis(q));
        hi = hash_flat(hi, iota.Z_eq_minus_basis(q));
        hi = hash_flat(hi, iota.B_eq_plus_basis(q));
        hi = hash_flat(hi, iota.B_eq_minus_basis(q));
        hi = hash_flat(hi, iota.H_plus_basis(q));
        hi = hash_flat(hi, iota.H_minus_basis(q));
    }
    out.iota = hi;
    out.dimHplus1 = iota.q_max() >= 1 ? iota.dim_H_plus(1) : 0;
    out.dimHminus1 = iota.q_max() >= 1 ? iota.dim_H_minus(1) : 0;

    // The mapping cone of the boundary restriction (compute() runs its certificate's verify()).
    hecke::SL2ZBoundaryPair pair(&coh, K);
    const uint64_t rc0 = engine::tape::host_elim_hook_regions();
    const std::unique_ptr<EquivariantMappingCone> cone = pair.make_cone();
    out.regions_cone = engine::tape::host_elim_hook_regions() - rc0;
    uint64_t hc = cone->golden_digest();
    for (uint32_t q = 0; q <= cone->q_max(); ++q) {
        hc = hash_block(hc, cone->Z_block(q));
        hc = hash_block(hc, cone->B_block(q));
    }
    out.cone = hc;
    out.dimHc1 = cone->q_max() >= 1 ? cone->dim_Hc(1) : 0;
    return out;
}

} // namespace

uint32_t HostElimThreadDeterminismValidation() {
    const auto t0 = std::chrono::steady_clock::now();
    std::cout << "=== HostElimThreadDeterminismValidation (hecke-extensions slot 23, PERF-P2) ===" << std::endl;
    std::cout << "    the cone / iota^* host eliminations under engine::tape::ScopedHostElimHook on consumer pools of 1, 4, 16"
                 " workers: every value identical, the hook taken (regions > 0) wherever the pool has > 1 worker." << std::endl;
    g_checks = 0;
    g_failures = 0;
    const bool hook_on = engine::tape::host_elim_hook_enabled();
    const std::vector<uint32_t> pools = { 1, 4, 16 };
    uint64_t direct_digest = 0, cone_digest = 0, iota_digest = 0, tall_digest = 0;
    uint64_t regions_total = 0;

    try {
        // ---------------------------------------------------------------- D: the direct arms
        // (audit L2) the tall case (m >= 1024) exercises base_panel's tiled pivot search (rows_pass) and its cross-tile
        // firsts[] minimum; it is folded into its own digest (tall=) so direct= keeps its pre-repair value.
        struct DCase { uint32_t m, n, p; uint64_t seed; bool tall; };
        const std::vector<DCase> dcases = { { 700, 900, 13u, 1, false }, { 640, 520, 134217757u, 2, false }, { 512, 512, 691u, 3, false },
                                                  { 1600, 300, 13u, 4, true } };
        uint64_t th = engine::kFnvOffset64;
        uint64_t dh = engine::kFnvOffset64;
        for (const DCase& c : dcases) {
            const DenseBuf<uint32_t> A = test_matrix(c.m, c.n, c.p, c.seed);
            const uint64_t ref = direct_arm(A, c.p);   // no hook on this thread: the serial route
            if (c.tall) th = engine::FnvU64(th, ref); else dh = engine::FnvU64(dh, ref);
            uint64_t regions_first_multi = 0;   // the region count at the first pool > 1: tiling must not depend on the pool size
            for (uint32_t w : pools) {
                ThreadPool pool(w);
                ScopedConsumerPool sp(&pool);
                const uint64_t r0 = engine::tape::host_elim_hook_regions();
                uint64_t got = 0;
                {
                    ScopedHostElimHook hook;
                    got = direct_arm(A, c.p);
                }
                const uint64_t regions = engine::tape::host_elim_hook_regions() - r0;
                regions_total += regions;
                const std::string at = " (" + std::to_string(c.m) + "x" + std::to_string(c.n) + " mod " + std::to_string(c.p) +
                                       ", pool " + std::to_string(w) + ")";
                Check(got == ref, "D: rref / column_rank_profile / rref_with_transform under the hook == hook-free" + at);
                if (hook_on && w > 1) Check(regions > 0, "D: the hook ran (regions > 0)" + at);
                else Check(regions == 0, "D: no hook region with one worker or under ENGINE_HOST_ELIM_HOOK=0" + at);
                if (w > 1) {
                    if (regions_first_multi == 0) regions_first_multi = regions;
                    else Check(regions == regions_first_multi, "D: the tiling (region count) does not depend on the pool size" + at);
                }
                std::cout << "    D " << (c.tall ? "(tall) " : "") << c.m << "x" << c.n << " mod " << c.p << " pool=" << w << " digest=" << hex(got)
                          << " regions=" << regions << std::endl;
            }
        }
        direct_digest = dh;

        // ---------------------------------------------------------------- C / I: the cone and iota^* at (n, ell) cells
        const SimplicialComplex K = CreateSL2ZFundamentalDomainTriangulation();
        const hecke::SL2ZTessellationTables model(K);
        struct PCase { uint32_t n, ell; };
        const std::vector<PCase> pcases = { { 64, 13u }, { 96, 691u }, { 64, 134217757u } };
        uint64_t ch = engine::kFnvOffset64, ih = engine::kFnvOffset64;
        for (const PCase& c : pcases) {
            PipelineDigests ref;
            for (size_t k = 0; k < pools.size(); ++k) {
                const uint32_t w = pools[k];
                ThreadPool pool(w);
                ScopedConsumerPool sp(&pool);
                const uint64_t r0 = engine::tape::host_elim_hook_regions();
                const PipelineDigests got = pipeline_cell(K, model, c.n, c.ell);
                const uint64_t regions = engine::tape::host_elim_hook_regions() - r0;
                regions_total += regions;
                const std::string at = " at (" + std::to_string(c.n) + ", " + std::to_string(c.ell) + "), pool " + std::to_string(w);
                if (k == 0) ref = got;
                Check(got.cone == ref.cone, "C: cone digest (golden objects + Z / B) == the pool-1 run" + at);
                Check(got.iota == ref.iota, "I: iota^* +- bases digest == the pool-1 run" + at);
                // (audit L1) the cone's and iota^*'s regions are gated SEPARATELY, so dropping either guard is caught.
                if (hook_on && w > 1) {
                    Check(got.regions_cone > 0, "C: the cone's hook ran (cone regions > 0)" + at);
                    Check(got.regions_iota > 0, "I: iota^*'s hook ran (iota regions > 0)" + at);
                } else {
                    Check(regions == 0, "C/I: no hook region with one worker or under ENGINE_HOST_ELIM_HOOK=0" + at);
                }
                std::cout << "    C/I (" << c.n << ", " << c.ell << ") pool=" << w << " cone=" << hex(got.cone)
                          << " iota=" << hex(got.iota) << " dim_Hc1=" << got.dimHc1 << " dim_H1+-=" << got.dimHplus1 << "/"
                          << got.dimHminus1 << " regions=" << regions << " (cone " << got.regions_cone << ", iota "
                          << got.regions_iota << ")" << std::endl;
            }
            ch = engine::FnvU64(ch, ref.cone);
            ih = engine::FnvU64(ih, ref.iota);
        }
        cone_digest = ch;
        iota_digest = ih;
        tall_digest = th;

        // ---------------------------------------------------------------- L3 (audit): guard semantics, on a 4-worker pool
        ThreadPool pool4(4);
        ScopedConsumerPool sp4(&pool4);
        const bool expect = hook_on;   // under the rollback no guard installs anything
        // (B) inheritance through run_concurrently: none from an unhooked caller, all from a hooked one.
        auto census = [&]() {
            std::vector<int> seen(6, 0);
            std::vector<std::function<void()>> tasks;
            for (size_t i = 0; i < seen.size(); ++i)
                tasks.push_back([&seen, i] {
                    seen[i] = (engine::tape::host_elim_hook_active() && engine::elim::parallel_hook() != nullptr) ? 1 : 0;
                });
            engine::tape::run_concurrently(tasks);
            int s = 0;
            for (int v : seen) s += v;
            return s;
        };
        const int unhooked = census();
        int hooked = 0;
        {
            ScopedHostElimHook hook;
            hooked = census();
            Check(engine::tape::host_elim_hook_active() == expect, "B: the caller's hook survives run_concurrently");
        }
        Check(unhooked == 0, "B: no task thread of an unhooked caller carries a hook");
        Check(hooked == (expect ? 6 : 0), "B: every task thread of a hooked caller carries its own hook");
        Check(!engine::tape::host_elim_hook_active() && engine::elim::parallel_hook() == nullptr, "B: the caller is restored");
        // (C) exceptions through a hook region and through a hooked run_concurrently task; nesting; precedence (L4).
        const DenseBuf<uint32_t> A = test_matrix(700, 900, 13u, 1);
        const uint64_t refA = direct_arm(A, 13u);
        uint32_t caught = 0;
        {
            ScopedHostElimHook outer;
            if (expect) {
                try {
                    (*engine::elim::parallel_hook())(16, [](uint32_t t) {
                        if (t == 3 || t == 11) throw std::runtime_error("P2 forced throw in a hook task");
                    });
                } catch (const std::runtime_error&) { ++caught; }
                Check(engine::elim::parallel_hook() != nullptr, "C: the hook survives an exception in its region");
            }
            try {
                std::vector<std::function<void()>> tasks;
                tasks.push_back([&] { (void)direct_arm(A, 13u); });
                tasks.push_back([] { throw std::runtime_error("P2 forced throw in a hooked task"); });
                tasks.push_back([&] { (void)direct_arm(A, 13u); });
                engine::tape::run_concurrently(tasks);
            } catch (const std::runtime_error&) { ++caught; }
            Check(direct_arm(A, 13u) == refA, "C: the pool is sound after the exceptions (direct arm bit-equal)");
            {
                ScopedHostElimHook inner;
                Check(!inner.installed(), "C: a nested guard installs nothing");
            }
            Check(engine::tape::host_elim_hook_active() == expect && (engine::elim::parallel_hook() != nullptr) == expect,
                  "C: the outer guard survives the nested one");
        }
        Check(caught == (expect ? 2u : 1u), "C: every forced exception reached its caller");
        Check(!engine::tape::host_elim_hook_active() && engine::elim::parallel_hook() == nullptr, "C: restored after unwinding");
        {
            // (L4) a hook already in effect (here a foreign serial one) is never replaced.
            const engine::elim::ParallelFor foreign = [](uint32_t nt, const std::function<void(uint32_t)>& fn) {
                for (uint32_t t = 0; t < nt; ++t) fn(t);
            };
            const engine::elim::ScopedParallelHook fh(&foreign);
            ScopedHostElimHook g;
            Check(!g.installed() && engine::elim::parallel_hook() == &foreign, "C: precedence -- a foreign hook is not replaced");
        }
        // (D) concurrent hooked callers: 4 cells at once through run_concurrently, against the sequential reference.
        {
            const PipelineDigests ref = pipeline_cell(K, model, 64, 13u);
            std::vector<PipelineDigests> got(4);
            std::vector<std::function<void()>> tasks;
            for (size_t i = 0; i < got.size(); ++i)
                tasks.push_back([&, i] { got[i] = pipeline_cell(K, model, 64, 13u); });
            engine::tape::run_concurrently(tasks);
            uint32_t bad = 0;
            for (const PipelineDigests& g : got) bad += (g.cone != ref.cone || g.iota != ref.iota) ? 1u : 0u;
            Check(bad == 0, "D: 4 concurrent cells (64, 13) == the sequential run");
            std::cout << "    L3 B inherit " << unhooked << "/6 -> " << hooked << "/6; C caught=" << caught
                      << "; D concurrent cells 4, mismatches " << bad << std::endl;
        }
    } catch (const std::exception& e) {
        Check(false, std::string("exception: ") + e.what());
    }

    const double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    std::cout << "EFFECTIVE HostElimThreadDeterminismValidation arm=" << (hook_on ? "hook" : "rollback")
              << " pools=1,4,16 regions=" << regions_total << std::endl;
    std::cout << "HOST_ELIM_DET direct=" << hex(direct_digest) << " tall=" << hex(tall_digest) << " cone=" << hex(cone_digest) << " iota=" << hex(iota_digest)
              << " checks=" << g_checks << " failing=" << g_failures << " verdict=" << (g_failures == 0 ? "OK" : "FAIL")
              << " (" << std::fixed << std::setprecision(1) << secs << " s)" << std::endl;
    std::cout << "=== HostElimThreadDeterminismValidation " << (g_failures == 0 ? "PASSED" : "FAILED") << " ===" << std::endl;
    return g_failures;
}
