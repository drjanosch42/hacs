// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// EngineDeterminismValidation.cpp -- see the header.
//
// The plan's determinism guarantee (D4) says any thread count or batching
// yields bit-identical buffers.  This validation asserts it on the SHIPPED
// pipeline, not on synthetic programs: the golden harness's own extraction
// code recomputes every golden object of a few (n_sym, ell) cells --
// cohomology bases and projectors, Theta_flat / Theta_eq / A^{(q)} for
// p in {2, 3, 5}, r in {1, 2}, the family routes, the decompositions and the
// reports -- under each engine setting, and (a) every record must equal the
// checked-in reference (tier A and tier B both gating), (b) the digest over
// all records must be the same in every arm.  A warm program cache is kept
// across the arms deliberately: from the second arm on, most programs are
// REPLAYS of programs compiled under another pool size (the batching arms
// compile their own, the key carries the options fingerprint).
#include "EngineDeterminismValidation.h"

#include <chrono>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include "../Engine/EngineRuntime.h"
#include "../Engine/ProgramCache.h"
#include "../Engine/TapeElim.h"
#include "../Engine/ThreadPool.h"
#include "Golden/GoldenHarness.h"

uint32_t EngineDeterminismValidation() {
    using namespace engine::tape;
    std::cout << "================= EngineDeterminismValidation =================\n";
    // Small / medium cells of the golden grid (n_sym x ell), chosen to cover
    // the regimes (ell = 2, a mid prime, ell ~ 2^27), even and odd n, the
    // (24, 5) Dickson locus, and both Hecke primes' patterns.
    const std::vector<std::pair<uint32_t, uint32_t>> cells = {{4, 13}, {10, 134217757}, {16, 691}, {22, 3}};
    struct Arm { uint32_t threads; bool batch; };
    std::vector<Arm> arms;
    for (uint32_t t : {1u, 4u, 16u, 32u})
        for (bool b : {true, false}) arms.push_back({t, b});

    uint32_t failures = 0;
    uint64_t digest0 = 0;
    size_t records0 = 0;
    ProgramCache<uint32_t>::global().clear();
    const auto t_start = std::chrono::steady_clock::now();
    for (size_t a = 0; a < arms.size(); ++a) {
        const Arm& arm = arms[a];
        ThreadPool pool(arm.threads);
        ScopedConsumerPool sp(&pool);
        PassOptions o = PassOptions::from_env();
        o.batch = arm.batch;
        ScopedPassOptions so(o);
        const auto t0 = std::chrono::steady_clock::now();
        const auto stats_before = ProgramCache<uint32_t>::global().stats();
        const TapeElimStats elim_before = tape_elim_totals();
        uint64_t digest = 0;
        size_t records = 0;
        uint32_t fails = 0;
        try {
            fails = GoldenSubset32(cells, digest, records, /*quiet=*/true);
        } catch (const std::exception& e) {
            std::cout << "  arm threads=" << arm.threads << " batch=" << arm.batch << ": EXCEPTION " << e.what() << "\n";
            ++failures;
            continue;
        }
        const auto stats_after = ProgramCache<uint32_t>::global().stats();
        const TapeElimStats elim_after = tape_elim_totals();
        const double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        if (a == 0) { digest0 = digest; records0 = records; }
        const bool same = (digest == digest0) && (records == records0);
        if (fails != 0 || !same) ++failures;
        std::cout << "  threads=" << std::setw(2) << arm.threads << " batch=" << arm.batch << ": " << records
                  << " golden records, " << fails << " differ from the reference, digest 0x" << std::hex << digest
                  << std::dec << (same ? " == arm 0" : " != arm 0 (FAIL)") << ", program cache hits +"
                  << (stats_after.hits - stats_before.hits) << " misses +" << (stats_after.misses - stats_before.misses)
                  << ", " << std::fixed << std::setprecision(2) << secs << " s\n";
        // (hecke-extensions slot 9, WP-6) non-vacuity: what the elimination
        // driver's per-panel programs were compiled under in this arm.  Since
        // WP-6 the driver reads consumer_pass_options(), so a `batch=0` arm
        // compiles every panel program with batch = false (before WP-6: none).
        std::cout << "    elim panel programs +" << (elim_after.programs - elim_before.programs)
                  << ": compiled batch=0 +" << (elim_after.batch_off_programs - elim_before.batch_off_programs)
                  << ", batched +" << (elim_after.batched_programs - elim_before.batched_programs) << "\n";
    }
    const double total = std::chrono::duration<double>(std::chrono::steady_clock::now() - t_start).count();
    std::cout << "EngineDeterminismValidation: " << arms.size() << " arms x " << cells.size() << " cells, " << records0
              << " records each, " << failures << " failing arms, " << std::fixed << std::setprecision(1) << total << " s\n";
    return failures;
}
