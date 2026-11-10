// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// ===========================================================================
// hecke-extensions slot 12, WP-6 -- HeckeCensusCellListValidation (`fast`).
//
// The census cell list (HeckeCensusCellList.h), the cell-list option
// HECKE_CENSUS_SPLIT_CELLS=<file> of HeckeCensusSplitValidation's arm (D):
//   (A) edition 1's grid: CensusEdition1Grid() is exactly the 406 cells arm
//       (D) ran in slot 11 (ell in {5, 7, 11, 13}, even n <= 200, then
//       (10, 691), (198, 691) at budget_mb=8192), in that order.
//   (B) a well-formed list parses in file order, comments / blank lines
//       ignored, the optional budget field read; the W2 list shipped with
//       census edition 2 (112 cells: ell in {7, 11}, even n = 190..300)
//       has the grid's shape.
//   (C) refusals, each with its line number: a missing field, a fourth
//       field, a non-decimal field, odd n, ell not a prime >= 5, a
//       duplicate cell, an empty list, a missing file.
// Pure parsing; no pipeline run, no file written.
// ===========================================================================

#include "HeckeCensusCellList.h"
#include "../Util/ValidationCheck.h"

#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

using namespace hecke_ext;

namespace {

oc_util::Checker Check("  *** FAIL [census-cell-list] ");

void Refused(const std::string& text, const std::string& want_prefix, const std::string& what) {
    std::vector<CensusCell> out;
    std::string err;
    const bool ok = ParseCensusCellList(text, out, err);
    Check(!ok, what + ": refused");
    Check(out.empty(), what + ": no cell returned");
    Check(err.rfind(want_prefix, 0) == 0, what + ": error `" + err + "` starts with `" + want_prefix + "`");
}

} // namespace

uint32_t HeckeCensusCellListValidation() {
    Check.reset();
    std::cout << "=== HeckeCensusCellListValidation (slot-12 WP-6; the census cell-list option) ===\n";

    // (A) edition 1's grid
    {
        const std::vector<CensusCell> g = CensusEdition1Grid();
        Check(g.size() == 406, "edition-1 grid: 406 cells (" + std::to_string(g.size()) + ")");
        size_t k = 0;
        bool order = true;
        for (uint32_t ell : { 5u, 7u, 11u, 13u })
            for (uint32_t n = 0; n <= 200; n += 2, ++k)
                order = order && k < g.size() && g[k].n == n && g[k].ell == ell && g[k].u_ell_budget_mb == 0;
        Check(order, "edition-1 grid: ell in {5,7,11,13} x even n <= 200, in order, default u_ell");
        Check(g.size() == 406 && g[404].n == 10 && g[404].ell == 691 && g[404].u_ell_budget_mb == 0,
              "edition-1 grid: (10, 691) at the default u_ell");
        Check(g.size() == 406 && g[405].n == 198 && g[405].ell == 691 && g[405].u_ell_budget_mb == 8192,
              "edition-1 grid: (198, 691) at budget_mb=8192 (as ext_u691)");
    }

    // (B) a well-formed list
    {
        const std::string text =
            "# census edition 2 -- test list\n"
            "\n"
            "190 7\n"
            "  300   11   # trailing comment\n"
            "198 691 8192\n"
            "10\t691\t0\n";
        std::vector<CensusCell> out;
        std::string err;
        const bool ok = ParseCensusCellList(text, out, err);
        Check(ok, "well-formed list parses (" + err + ")");
        Check(out.size() == 4, "four cells");
        if (out.size() == 4) {
            Check(out[0].n == 190 && out[0].ell == 7 && out[0].u_ell_budget_mb == 0, "cell 1 = (190, 7)");
            Check(out[1].n == 300 && out[1].ell == 11 && out[1].u_ell_budget_mb == 0, "cell 2 = (300, 11), comment stripped");
            Check(out[2].n == 198 && out[2].ell == 691 && out[2].u_ell_budget_mb == 8192, "cell 3 = (198, 691) budget 8192");
            Check(out[3].n == 10 && out[3].ell == 691 && out[3].u_ell_budget_mb == 0, "cell 4 = (10, 691), tabs");
        }
        // The W2 grid (ext_census_l7l11): ell in {7, 11}, even n = 190..300.
        std::string w2;
        for (uint32_t ell : { 7u, 11u })
            for (uint32_t n = 190; n <= 300; n += 2) w2 += std::to_string(n) + " " + std::to_string(ell) + "\n";
        const bool ok2 = ParseCensusCellList(w2, out, err);
        Check(ok2 && out.size() == 112, "the W2 grid: 112 cells (" + std::to_string(out.size()) + ")");
    }

    // (C) refusals
    Refused("190\n", "line 1:", "one field");
    Refused("# c\n190 7 0 1\n", "line 2:", "four fields");
    Refused("190 7x\n", "line 1:", "non-decimal ell");
    Refused("-2 7\n", "line 1:", "negative n");
    Refused("190 7 big\n", "line 1:", "non-decimal budget");
    Refused("191 7\n", "line 1:", "odd n");
    Refused("190 3\n", "line 1:", "ell = 3");
    Refused("190 9\n", "line 1:", "ell = 9 not prime");
    Refused("190 7\n192 7\n190 7\n", "line 3:", "duplicate cell");
    Refused("1234567890 7\n", "line 1:", "an overlong field");
    Refused("# only comments\n\n", "the cell list holds no cell", "empty list");
    {
        std::vector<CensusCell> out;
        std::string err;
        const bool ok = ReadCensusCellList("/nonexistent/wp6/census_cells.txt", out, err);
        Check(!ok && out.empty() && err.rfind("cannot read", 0) == 0, "a missing file is refused (" + err + ")");
    }

    std::cout << "HeckeCensusCellListValidation: " << Check.checks << " checks, " << Check.fails << " failing\n";
    return Check.fails == 0 ? 0 : 1;
}
