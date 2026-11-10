// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// HeckeExtDeterminismValidation.cpp -- see the header.
//
// [X] = docs/implementation/hecke_extensions_implementation_plan.md v1.8
// Sec. 5 (this validation), Sec. 2.4 risk R10 ("order dependence creeping in
// -- hash containers, thread-dependent accumulation").
//
// THE CLAIM.  A `hecke_ext/v2` record is a mathematical object, so its TEXT is
// a function of (n_sym, ell, the mathematical settings) and of nothing else --
// not of ENGINE_THREADS, not of ENGINE_BATCH, not of whether the tape ran a
// node on the CPU or the GPU.  The record is the campaign's published output
// and WP-3 will hash parts of it into GoldenReference.inc, so this is the
// statement that has to hold before that emission is meaningful.
//
// HOW IT IS CHECKED.  Exactly EngineDeterminismValidation's shape, one level
// up: for each arm (a thread count, a batching setting, a device policy) the
// whole cell is recomputed through HeckeExt/HeckeExtDriver and the FNV-1a-64
// of BuildHeckeExtReportJson(record) is compared with arm 0's.  The record
// text is the digest input rather than a hand-picked list of invariants,
// because a hand-picked list is exactly what an order-dependence bug slips
// past.
//
// The `commit` field IS excluded (hecke-extensions slot 6, WP-6 / E2
// close-out): it is metadata baked in from `git rev-parse HEAD` at CMake
// CONFIGURE time (CMakeLists.txt), so it is identical across arms of one
// BINARY but moves on every re-configure at a different HEAD, a
// documents-only commit included -- which is exactly what made the printed
// digest non-reproducible across checkouts (slot-5 WP-3 finding 3). It is
// redacted with the golden harness's own `RedactJsonKeys`, exposed as
// `GoldenRedactJsonKeys` (Golden/GoldenHarness.h) so there is one definition
// of the redaction, not a second hand-copied one; timings are NOT in the
// record ([E] Sec. 7).
//
// The top-level `settings` block IS REDUCED TO ITS FINGERPRINT (hecke-
// extensions slot 7, WP-2; owner decision 2, 2026-09-23): the same
// path-scoped recipe the golden ext objects take, through the ONE definition
// `GoldenReduceSettingsToFingerprint` (Golden/GoldenHarness.h).  A key
// registered at its default therefore no longer moves this digest; a setting
// in force still does, through the fingerprint.  The printed digest moved
// ONCE, at that change (0xfb71e225e4185be7 -> the value in the slot-7 golden
// ledger), and the heavy-cell digest with it.
#include "HeckeExtDeterminismValidation.h"

#include <chrono>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include "Golden/GoldenHarness.h"
#include "HeckeExt/HeckeExtDriver.h"
#include "HeckeExt/HeckeExtRecord.h"
#include "HeckeExt/HeckeExtReport.h"
#include "Hecke/HeckeOperatorOnK.h"   // (PERF-VF audit S3) ScopedFieldDiet

#include "../ModArithmetic/Fnv.h"
#include "../Engine/DeviceBackend.h"
#include "../Engine/EngineRuntime.h"
#include "../Engine/Passes.h"
#include "../Engine/Program.h"
#include "../Engine/ProgramCache.h"
#include "../Engine/TapeElim.h"
#include "../Engine/ThreadPool.h"

namespace {

struct Cell {
    uint32_t n_sym;
    uint32_t ell;
    bool     attachments;   // with_cone + with_sign_split (the v2 objects)
};

struct Arm {
    uint32_t threads;
    bool     batch;
    engine::tape::DevicePolicy device;
    const char* device_name;
    bool     heavy_cell;    // include the expensive (128, 5) cell
};

// `commit` redacted and `settings` reduced to its fingerprint -- see the file
// header comment.  The record text is whitespace-free as written.
std::string DigestText(const hecke_ext::HeckeExtRecord& rec) {
    return GoldenRedactJsonKeys(
        GoldenReduceSettingsToFingerprint(hecke_ext::BuildHeckeExtReportJson(rec)), {"commit"});
}

uint64_t RecordDigest(const Cell& c) {
    hecke_ext::ExtDriverOptions opts;
    opts.with_cone = c.attachments;
    opts.with_sign_split = c.attachments;
    hecke_ext::HeckeExtDriver d(c.n_sym, c.ell, opts);
    d.run();
    uint64_t h = engine::FnvBytes(engine::kFnvOffset64, DigestText(d.record()));
    if (d.has_sign_records()) {
        h = engine::FnvBytes(h, DigestText(d.sign_record(true)));
        h = engine::FnvBytes(h, DigestText(d.sign_record(false)));
    }
    return h;
}

} // namespace

uint32_t HeckeExtDeterminismValidation() {
    using namespace engine::tape;
    std::cout << "=============== HeckeExtDeterminismValidation ===============\n";

    // [X] Sec. 5's cells: (10, 691) -- the Ramanujan cell, one local ideal
    // with a length-2 Loewy series; (34, 5) and (24, 7) -- two cells where the
    // Sturm sweep needs a prime outside S_0 and the naive bound B(k) is wrong;
    // and (128, 5), the large cell, on the cpu arms at threads {1, 16} only.
    const std::vector<Cell> cells = {{10, 691, true}, {34, 5, true}, {24, 7, true}};
    const std::vector<Cell> heavy = {{128, 5, false}};

    std::vector<Arm> arms;
    for (uint32_t t : {1u, 4u, 16u, 32u})
        for (bool b : {true, false})
            arms.push_back({t, b, DevicePolicy::Cpu, "cpu", (t == 1u || t == 16u) && b});
    const bool device_usable = engine::tape::DeviceBackend<uint32_t>::usable();
    if (device_usable) {
        for (uint32_t t : {1u, 16u}) arms.push_back({t, true, DevicePolicy::Gpu, "gpu", false});
    } else {
        std::cout << "  [SKIP] no usable CUDA device: the two `device = gpu` arms are not run\n";
    }

    uint32_t failures = 0;
    uint64_t digest0 = 0, heavy0 = 0;
    bool have0 = false, have_heavy0 = false;
    const auto t_start = std::chrono::steady_clock::now();

    for (size_t a = 0; a < arms.size(); ++a) {
        const Arm& arm = arms[a];
        ThreadPool pool(arm.threads);
        ScopedConsumerPool sp(&pool);
        PassOptions o = PassOptions::from_env();
        o.batch = arm.batch;
        o.policy = arm.device;
        ScopedPassOptions so(o);
        // (hecke-extensions slot 9, WP-6) non-vacuity instruments, diagnostic
        // only: a DeviceTally under an Inherit override (it changes no option)
        // counts every program the arm RUNS and its GPU-assigned nodes; the
        // cache size delta is the number of NEW slots the arm keyed; the
        // elimination totals say what the per-panel programs compiled under.
        DeviceTally tally;
        ScopedThreadDevice tally_scope(ThreadDevice::Inherit, &tally);
        const size_t slots_before = ProgramCache<uint32_t>::global().size();
        const auto cache_before = ProgramCache<uint32_t>::global().stats();
        const TapeElimStats elim_before = tape_elim_totals();

        const auto t0 = std::chrono::steady_clock::now();
        uint64_t digest = engine::kFnvOffset64;
        uint64_t hdigest = engine::kFnvOffset64;
        bool threw = false;
        try {
            for (const Cell& c : cells) digest = engine::FnvU64(digest, RecordDigest(c));
            if (arm.heavy_cell)
                for (const Cell& c : heavy) hdigest = engine::FnvU64(hdigest, RecordDigest(c));
        } catch (const std::exception& e) {
            std::cout << "  arm threads=" << arm.threads << " batch=" << arm.batch
                      << " device=" << arm.device_name << ": EXCEPTION " << e.what() << "\n";
            ++failures;
            threw = true;
        }
        const double secs =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        if (threw) continue;
        if (!have0) { digest0 = digest; have0 = true; }
        const bool same = (digest == digest0);
        bool heavy_same = true;
        if (arm.heavy_cell) {
            if (!have_heavy0) { heavy0 = hdigest; have_heavy0 = true; }
            heavy_same = (hdigest == heavy0);
        }
        if (!same || !heavy_same) ++failures;
        std::cout << "  threads=" << std::setw(2) << arm.threads << " batch=" << arm.batch
                  << " device=" << arm.device_name << ": " << cells.size()
                  << " cells, digest 0x" << std::hex << digest << std::dec
                  << (same ? " == arm 0" : " != arm 0 (FAIL)");
        if (arm.heavy_cell)
            std::cout << ", +(128,5) 0x" << std::hex << hdigest << std::dec
                      << (heavy_same ? " == arm 0" : " != arm 0 (FAIL)");
        std::cout << ", " << std::fixed << std::setprecision(2) << secs << " s\n";
        const auto cache_after = ProgramCache<uint32_t>::global().stats();
        const TapeElimStats elim_after = tape_elim_totals();
        std::cout << "    cache: new slots +" << (ProgramCache<uint32_t>::global().size() - slots_before)
                  << " hits +" << (cache_after.hits - cache_before.hits) << " misses +"
                  << (cache_after.misses - cache_before.misses) << "; programs run "
                  << tally.programs.load() << " (with GPU nodes " << tally.gpu_programs.load() << ", "
                  << tally.gpu_nodes.load() << " nodes); elim panel programs +"
                  << (elim_after.programs - elim_before.programs) << " (batch=0 +"
                  << (elim_after.batch_off_programs - elim_before.batch_off_programs) << ", batched +"
                  << (elim_after.batched_programs - elim_before.batched_programs) << ", device policy +"
                  << (elim_after.device_policy_programs - elim_before.device_policy_programs) << ", GPU +"
                  << (elim_after.gpu_programs - elim_before.gpu_programs) << " programs / +"
                  << (elim_after.gpu_nodes - elim_before.gpu_nodes) << " nodes)\n";
    }

    // (PERF-VF audit S3) The driver-level Stage-1 release.  Since PERF-VF (b2) it runs only from
    // max dim C_eq >= kFieldDietReleaseMinDimCeq (n >= 299), above every fast / standard ext cell, so
    // the post-release path (the anemic pool, the Sturm sweep's reuse, T_p read from the cached
    // A^{(q)}) would otherwise go unexercised by --all.  The Ramanujan cell (10, 691) is recomputed
    // with the floor forced to 0 (released) and its digest compared with the default run (not
    // released) on the default pool.  A separate check: it does not enter the digest printed below.
    uint32_t release_failures = 0;
    {
        const Cell c = cells[0];   // (10, 691) with cone + sign split
        const hecke::HeckeOperatorOptions::ScopedFieldDiet diet_on(true);
        const uint64_t r0 = hecke_ext::stage_one_releases();
        uint64_t d_default = 0, d_forced = 0;
        bool threw = false;
        try {
            d_default = RecordDigest(c);
            const uint64_t r1 = hecke_ext::stage_one_releases();
            {
                const hecke_ext::ScopedReleaseMinDimCeq force(0);
                d_forced = RecordDigest(c);
            }
            const uint64_t r2 = hecke_ext::stage_one_releases();
            const bool ok = r1 == r0 && r2 == r1 + 1 && d_forced == d_default &&
                            hecke_ext::field_diet_release_min_dim_ceq() == hecke_ext::kFieldDietReleaseMinDimCeq;
            if (!ok) ++release_failures;
            std::cout << "  STAGE1_RELEASE cell=(" << c.n_sym << "," << c.ell << ") default: releases +" << (r1 - r0)
                      << " digest 0x" << std::hex << d_default << std::dec << "; floor 0: releases +" << (r2 - r1)
                      << " digest 0x" << std::hex << d_forced << std::dec << (ok ? " (released == not released)" : " (FAIL)")
                      << "\n";
        } catch (const std::exception& e) {
            std::cout << "  STAGE1_RELEASE: EXCEPTION " << e.what() << "\n";
            threw = true;
        }
        if (threw) ++release_failures;
    }

    // (PERF-VF re-audit R3) a Stage-1 release failure counts in failing_arms (the digest is untouched).
    failures += release_failures;
    const double total =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - t_start).count();
    std::cout << "HECKE_EXT_DETERMINISM arms=" << arms.size() << " cells=" << cells.size()
              << " heavy_cells=" << heavy.size() << " device=" << (device_usable ? 1 : 0)
              << " digest=0x" << std::hex << digest0 << std::dec
              << " failing_arms=" << failures << " " << std::fixed << std::setprecision(1)
              << total << " s" << (release_failures ? " stage1_release=FAIL" : "") << "\n";
    return failures;
}
