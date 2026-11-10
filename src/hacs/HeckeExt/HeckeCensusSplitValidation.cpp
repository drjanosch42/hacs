// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// ===========================================================================
// hecke-extensions slot 11, WP-7 -- HeckeCensusSplitValidation (`standard`).
//
// [E] `sec:census` (d1), (d2), (d6), (e); slot-11 plan Sec. 3 "WP-7" scope 2.
//
//   (A) the benchmark (10, 691): the Kummer ideal is `mixed`, dim_E = 3,
//       rank_r1 = 1 ([E] sec:census (d1) "Expected at the benchmark",
//       cl:ramanujan), with every per-cell invariant.
//   (B) a grid at ell in {5, 7, 11, 13}: every per-cell invariant of (d1),
//       (d2) (sum dim_E = dim H^1, sum rank_r1 = rank r^1 = dim H^1 -
//       dim H^1_!, sum dim_plus = dim H^1_+), identity 6 kPassed and no
//       `identity_6_mismatch` row; each grid cell is one the published
//       census reads.
//   (C) the negative controls: the t_p fingerprint moves under a one-entry
//       perturbation; the verdict thresholds on hand-made ranks.
//   (D) under the read-only interlock HECKE_CENSUS_SPLIT_ALL=1 (NOT an
//       emitter; the HECKE_INTEGRAL_CAMPAIGN pattern): every census cell of
//       (d1) -- ell in {5, 7, 11, 13}, even n <= 200, and (10, 691),
//       (198, 691) (the latter under u_ell = budget_mb=8192, as ext_u691) --
//       with one CENSUS_SPLIT line per H^1 row, which
//       docs/matrix-engine/tools/congruence_census.py parses.  Without the
//       variable the arm prints its [SKIP] line.
//       Slot-12 WP-6: the CELL-LIST option HECKE_CENSUS_SPLIT_CELLS=<file>
//       (read-only interlock, not an emitter; HeckeCensusCellList.h's
//       format `<n> <ell> [<u_ell_budget_mb>]`) runs arm (D) over the listed
//       cells instead of edition 1's grid (census edition 2's W2 cells).
//       Without it arm (D) is edition 1's grid exactly (CensusEdition1Grid(),
//       the same 406 cells in the same order); at the defaults (neither
//       variable) no check count moves.  A malformed list is one failing
//       check and runs no cell.  PRECEDENCE (slot-12 WP-11, A12): when both
//       variables are set, HECKE_CENSUS_SPLIT_CELLS wins and the arm says so
//       ("HECKE_CENSUS_SPLIT_ALL=1 ignored").  The Glover reconciliation
//       (identity 6, CheckCell's second argument) is asserted exactly at
//       ell in {5, 7, 11, 13} -- the primes with an in-process Sigma(ell); at
//       edition 1's grid and the W2 list that is the old `ell != 691`, at any
//       other listed prime it is reported, not asserted.
// ===========================================================================

#include "HeckeCensusSplit.h"
#include "HeckeCensusCellList.h"
#include "../Util/ValidationCheck.h"


#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <map>
#include <string>
#include <vector>

using namespace hecke_ext;

namespace {

oc_util::Checker Check("  *** FAIL [census-split] ");

std::string Tag(uint32_t n, uint32_t ell) {
    return "(" + std::to_string(n) + "," + std::to_string(ell) + ")";
}

struct WeightData {
    bool ok = false;
    std::string why;
    std::vector<std::vector<ResidueCharacterMultiset>> w;
    std::vector<uint64_t> primes;
};

const WeightData& Weights(uint32_t ell) {
    static std::map<uint32_t, WeightData> cache;
    auto it = cache.find(ell);
    if (it != cache.end()) return it->second;
    WeightData d;
    d.ok = CensusGloverWeights(ell, d.w, d.primes, d.why);
    return cache.emplace(ell, d).first->second;
}

CensusSplitCell RunCell(uint32_t n, uint32_t ell, uint32_t budget_mb) {
    CensusSplitOptions o;
    o.u_ell_budget_mb = budget_mb;
    const bool glover = (ell == 5 || ell == 7 || ell == 11 || ell == 13);
    if (!glover) return ComputeCensusSplit(n, ell, o, nullptr, nullptr);
    const WeightData& w = Weights(ell);
    Check(w.ok, "ell=" + std::to_string(ell) + ": weight cells (" + w.why + ")");
    return ComputeCensusSplit(n, ell, o, w.ok ? &w.w : nullptr, w.ok ? &w.primes : nullptr);
}

void CheckCell(const CensusSplitCell& c, bool expect_glover) {
    const std::string tag = Tag(c.n, c.ell);
    for (const std::string& f : c.failures) Check(false, tag + ": " + f);
    Check(c.failures.empty(), tag + ": every per-cell invariant holds");
    if (expect_glover && c.dim_H1 > 0) {
        Check(c.glover_status == "evaluated", tag + ": Glover evaluated (" + c.glover_status + ")");
        Check(c.identity_code == 0, tag + ": IdentityCertificate::verify() = "
                                        + std::to_string(c.identity_code) + " (" + c.identity_detail + ")");
    }
    for (const CensusSplitRow& r : c.rows) {
        Check(r.verdict == "eisenstein" || r.verdict == "cuspidal" || r.verdict == "mixed",
              tag + ": a verdict token");
        Check(!r.identity6_mismatch, tag + ": ideal " + std::to_string(r.ideal_id) + " identity 6");
    }
}

void PrintCell(const CensusSplitCell& c) {
    std::cout << "    " << Tag(c.n, c.ell) << " dim H^1=" << c.dim_H1 << " rank r^1=" << c.rank_r1
              << " H^1_+=" << c.dim_H1_plus << " rows=" << c.rows.size() << " glover=" << c.glover_status
              << " (" << std::fixed << std::setprecision(2) << c.seconds << " s)\n";
    for (const CensusSplitRow& r : c.rows) std::cout << "      " << CensusSplitLine(c, r) << "\n";
}

} // namespace

uint32_t HeckeCensusSplitValidation() {
    Check.reset();
    std::cout << "=== HeckeCensusSplitValidation (slot-11 WP-7; [E] sec:census (d1), (d2), (d6), (e)) ===\n";

    // --- (A) the benchmark -------------------------------------------------
    std::cout << "-- [A] the benchmark (10, 691)\n";
    {
        const CensusSplitCell c = RunCell(10, 691, 0);
        PrintCell(c);
        CheckCell(c, false);
        Check(c.rows.size() == 1, "(10,691): one H^1 ideal");
        if (c.rows.size() == 1) {
            const CensusSplitRow& r = c.rows[0];
            Check(r.verdict == "mixed", "(10,691): the Kummer ideal is mixed (" + r.verdict + ")");
            Check(r.dim_E == 3, "(10,691): dim_E = 3");
            Check(r.rank_r1 == 1, "(10,691): rank_r1 = 1");
            Check(r.glover_token == "not_evaluated", "(10,691): Glover not evaluated at 691");
        }
    }

    // --- (B) the grid --------------------------------------------------------
    std::cout << "-- [B] the grid at ell in {5, 7, 11, 13}\n";
    const std::vector<std::pair<uint32_t, uint32_t>> grid = {
        {0, 5}, {24, 5}, {34, 5}, {30, 5}, {82, 7}, {56, 7}, {24, 7}, {22, 11}, {44, 11}, {12, 13}, {40, 13},
    };
    uint32_t mixed = 0, rows = 0;
    for (const auto& nc : grid) {
        const CensusSplitCell c = RunCell(nc.first, nc.second, 0);
        PrintCell(c);
        CheckCell(c, true);
        rows += static_cast<uint32_t>(c.rows.size());
        for (const CensusSplitRow& r : c.rows) mixed += (r.verdict == "mixed");
    }
    std::cout << "    grid: " << grid.size() << " cells, " << rows << " H^1 rows, " << mixed << " mixed\n";

    // --- (C) negative controls -------------------------------------------------
    std::cout << "-- [C] negative controls\n";
    {
        MatrixT<uint32_t> a(2, 2, 7);
        a.data[0][0] = 1; a.data[0][1] = 2; a.data[1][0] = 3; a.data[1][1] = 4;
        MatrixT<uint32_t> b = a;
        b.data[1][1] = 5;
        const uint64_t fa = CensusTpFingerprint({2}, {a});
        Check(fa == CensusTpFingerprint({2}, {a}), "fingerprint deterministic");
        Check(fa != CensusTpFingerprint({2}, {b}), "fingerprint moves under a one-entry perturbation");
        Check(fa != CensusTpFingerprint({3}, {a}), "fingerprint moves with the prime");
    }

    // --- (D) the interlocked full run --------------------------------------------
    const char* all = std::getenv("HECKE_CENSUS_SPLIT_ALL");
    const char* cells_file = std::getenv("HECKE_CENSUS_SPLIT_CELLS");
    const bool use_list = cells_file != nullptr && std::string(cells_file).size() > 0;
    if (use_list || (all != nullptr && std::string(all) == "1")) {
        std::vector<CensusCell> grid;
        bool grid_ok = true;
        if (use_list) {
            std::string err;
            grid_ok = ReadCensusCellList(cells_file, grid, err);
            std::cout << "-- [D] HECKE_CENSUS_SPLIT_CELLS=" << cells_file << ": " << (grid_ok ? std::to_string(grid.size()) + " listed cell(s)" : "REFUSED (" + err + ")") << "\n";
            if (all != nullptr && std::string(all) == "1")
                std::cout << "-- [D] HECKE_CENSUS_SPLIT_ALL=1 ignored: HECKE_CENSUS_SPLIT_CELLS takes precedence\n";
            Check(grid_ok, std::string("the cell list parses (") + err + ")");
        } else {
            std::cout << "-- [D] HECKE_CENSUS_SPLIT_ALL=1: every census cell of [E] sec:census (d1)\n";
            grid = CensusEdition1Grid();
        }
        uint32_t cells = 0, lines = 0, failing = 0;
        auto run = [&](uint32_t n, uint32_t ell, uint32_t budget) {
            const CensusSplitCell c = RunCell(n, ell, budget);
            CheckCell(c, ell == 5 || ell == 7 || ell == 11 || ell == 13);
            ++cells;
            if (!c.failures.empty()) ++failing;
            std::cout << "CENSUS_CELL n=" << n << " ell=" << ell << " dim_H1=" << c.dim_H1
                      << " rank_r1=" << c.rank_r1 << " dim_H1_plus=" << c.dim_H1_plus
                      << " dim_H1_minus=" << c.dim_H1_minus << " rows=" << c.rows.size()
                      << " glover=" << c.glover_status << " identity_code=" << c.identity_code
                      << " failures=" << c.failures.size() << " seconds=" << std::fixed
                      << std::setprecision(2) << c.seconds << "\n";
            for (const CensusSplitRow& r : c.rows) {
                std::cout << CensusSplitLine(c, r) << "\n";
                ++lines;
            }
            std::cout.flush();
        };
        if (grid_ok)
            for (const CensusCell& g : grid) run(g.n, g.ell, g.u_ell_budget_mb);
        std::cout << "    [D] " << cells << " cells, " << lines << " CENSUS_SPLIT lines, " << failing
                  << " failing cell(s)\n";
        std::cout << "EFFECTIVE HeckeCensusSplitValidation arm=D cells=" << cells << "\n";
    } else {
        std::cout << "-- [D] [SKIP] the full census run (set HECKE_CENSUS_SPLIT_ALL=1, or HECKE_CENSUS_SPLIT_CELLS=<file>; read-only)\n";
    }

    std::cout << "HeckeCensusSplitValidation: " << Check.checks << " checks, " << Check.fails << " failing\n";
    return Check.fails == 0 ? 0 : 1;
}
