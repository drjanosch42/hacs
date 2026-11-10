// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

#pragma once
#ifndef HECKE_EXT_CENSUS_CELL_LIST_H
#define HECKE_EXT_CENSUS_CELL_LIST_H

// ===========================================================================
// hecke-extensions slot 12, WP-6 -- THE CENSUS CELL LIST: the cell-list
// option of HeckeCensusSplitValidation's arm (D) (slot-12 plan Sec. 2
// "WP-6"; slot-11 WP-7 hard-coded edition 1's grid, "a cell-list option is
// slot 12's").
//
// FORMAT (one cell per line; `#` starts a comment; blank lines ignored):
//     <n> <ell> [<u_ell_budget_mb>]
// `n` even (odd n vanishes on the model: the -I / odd-n gap), `ell` a prime
// >= 5 (the census's split is taken at ell >= 5; [E] `sec:census` (d1)),
// the optional third field the `u_ell = budget_mb=<value>` of the cell
// (0 / absent = the setting in force, as every edition-1 cell except
// (198, 691)).  A cell listed twice, a malformed or out-of-range field, or
// a fourth field is refused with the line number -- the arm then runs no
// cell (a partial census is never produced from a malformed list).
//
// Pure STL; no pipeline dependency.  The cell list is read-only input: it
// selects cells, it never changes a value (HECKE_CENSUS_SPLIT_CELLS is a
// read-only interlock like HECKE_CENSUS_SPLIT_ALL, not an emitter).
// ===========================================================================

#include <cstdint>
#include <string>
#include <vector>

namespace hecke_ext {

struct CensusCell {
    uint32_t n = 0, ell = 0;
    uint32_t u_ell_budget_mb = 0;   // 0 = the u_ell setting in force
};

// Parses the text of a cell list.  True on success (`out` in file order);
// false with `error` = "line <k>: <why>" on the first refused line (or
// "the cell list holds no cell"), `out` cleared.
bool ParseCensusCellList(const std::string& text, std::vector<CensusCell>& out, std::string& error);

// Reads `path` and parses it; a missing / unreadable file is an error.
bool ReadCensusCellList(const std::string& path, std::vector<CensusCell>& out, std::string& error);

// Edition 1's grid, exactly as slot-11 WP-7's arm (D) ran it: ell in
// {5, 7, 11, 13}, even n = 0..200, then (10, 691) and (198, 691) under
// u_ell = budget_mb=8192 -- 406 cells, in that order.
std::vector<CensusCell> CensusEdition1Grid();

} // namespace hecke_ext

#endif // HECKE_EXT_CENSUS_CELL_LIST_H
