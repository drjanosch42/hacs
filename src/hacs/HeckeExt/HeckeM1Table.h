// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

#pragma once
#ifndef HECKE_EXT_M1_TABLE_H
#define HECKE_EXT_M1_TABLE_H

// ===========================================================================
// hecke-extensions slot 5, WP-8 -- M1, THE PROGRAMME'S FIRST OWNER-VISIBLE
// DELIVERABLE, as one row per (cell, group, maximal ideal).
//
// [X] = docs/implementation/hecke_extensions_implementation_plan.md v1.8
// Sec. 6.2 E2 (M1) and Sec. 1.3.1 R3.2 (the FIXED reason list); [E] =
// docs/latex/hecke_extensions.tex v1.5.2 Sec. 8 (`sec:tests`, the case list)
// and Sec. 7 (the record); the slot-5 plan Sec. 3 WP-8 scope item 5.
//
// THE M1 STATEMENT, verbatim from the slot-5 plan:
//
//   "for EVERY case of [E] Sec. 8 that lives in H^0/H^1, a record on
//    X_1 (+) X_4, each ideal `final = true` or `final = false` WITH A REASON
//    FROM THE FIXED LIST of [X] Sec. 1.3.1 R3.2, and B'(k, ell) on every row.
//    Dickson-type ideals are `sturm_heuristic` by [E] Claim 2.10, i.e.
//    published `final = false`, not certified -- that is [X] Sec. 6.2 E2's own
//    wording and it is the expected outcome, not a defect."
//
// WHY A UNIT AND NOT THREE COPIES.  M1 is stated once and produced by three
// validations (Kummer, small-ell, separation), each over a different slice of
// [E] Sec. 8.  Defining "an M1 row" and "a well-formed M1 row" in one place is
// what makes the three tables comparable and the deliverable a single object;
// three private copies of the same 60 lines would make "the M1 table" three
// different things.  Nothing here computes: every field is read off a
// COMPLETED `HeckeExtRecord` (plus, for the B' column, the pipeline-free
// `HeckeSturmBound` -- which is exactly how the driver itself obtains it).
//
// PLACEMENT.  HeckeExt/, namespace hecke_ext, `MatrixT<uint32_t>`-free (it
// stores no matrix).  No Console/, no Cuda/, no Engine/, no BlockFlatMatrix.h,
// no `unordered_`.  It includes no Matrix.h directly and therefore needs no
// entry in check_matrix_includes.sh's allow-list.
// ===========================================================================

#include "HeckeExtRecord.h"

#include <cstdint>
#include <string>
#include <vector>

namespace hecke_ext {

// ---------------------------------------------------------------------------
// One M1 row.
// ---------------------------------------------------------------------------
struct M1Row {
    // The cell.
    uint32_t    n      = 0;
    uint64_t    ell    = 0;
    uint32_t    weight = 0;             // k = n + 2
    std::string case_label;             // the [E] Sec. 8 item this cell is from

    // The cell-level columns, repeated on every row of the cell so that a row
    // is self-contained (a table is read row by row).
    uint32_t    dim_H0 = 0;
    uint32_t    dim_H1 = 0;
    bool        u_ell_included = false;
    std::string u_ell_reason;           // within_budget | u_ell_over_budget | off

    // B'(k, ell), the CELL's Sturm bound -- "B'(k, ell) on every row".
    // `has_B_prime == false` means the bound does not exist below the cap, in
    // which case `B_prime_note` says which of the two it is.
    bool        has_B_prime = false;
    uint32_t    B_prime     = 0;
    uint32_t    B_prime_cap = 0;
    std::string B_prime_note;           // found | none_at_cap | not_applicable

    // The ideal.
    std::string group;                  // "H0" = X_1, "H1" = X_4
    uint32_t    ideal_id       = 0;
    uint32_t    dim            = 0;
    uint32_t    residue_degree = 0;
    bool        local          = false;
    uint32_t    algebra_dim    = 0;
    uint32_t    embdim         = 0;
    uint32_t    nilpotency     = 0;
    bool        gorenstein     = false;
    std::vector<uint32_t> loewy;
    std::vector<uint32_t> socle;
    bool        cyclic         = false;

    // Generation.
    std::string certificate;            // cyclic | commutant | sturm(B') | dickson | boundary | unavailable
    std::string route_a, route_b, route_c;
    std::string route_d0 = "not_run", route_d1 = "not_run";   // (slot-12 FIN, hecke_ext/v4)

    // The verdict.
    std::string final_value;            // "true" | "false" | "heuristic"
    std::string final_reason;           // "" iff final_value == "true"
};

// ---------------------------------------------------------------------------
// The rows of one completed record, in the record's own ideal order.
// ---------------------------------------------------------------------------
//
// `case_label` is the caller's -- the [E] Sec. 8 item the cell belongs to --
// and is copied onto every row.  The B' columns are computed ONCE per call
// with `HeckeSturmBound` (`cap == 0` = the default 12*B(k)), which is the same
// object and the same cap the driver's Stage 4(c) uses; a weight with
// `dim M_k == 0` (n = 0, odd n, k < 4) gets `not_applicable`.
std::vector<M1Row> M1RowsFromRecord(const HeckeExtRecord& record,
                                    const std::string& case_label);

// ---------------------------------------------------------------------------
// Well-formedness -- the M1 statement itself, as a predicate.
// ---------------------------------------------------------------------------
//
// 0 on success, else the first failing clause:
//   1  the group is neither X_1 ("H0") nor X_4 ("H1")
//   2  `final_value` is not one of "true" / "false" / "heuristic"
//   3  the reason is not well formed: non-empty on a `true`, empty on a
//      non-`true`, or -- the clause that matters -- NOT A MEMBER OF THE FIXED
//      LIST of [X] Sec. 1.3.1 R3.2 (hecke_ext::IsFinalReason)
//   4  no B' column at all: neither a bound nor a note saying why not
//   5  the certificate string is not one of the four enumerated values, or a
//      certificate is claimed while every route says `not_run`
//   6  a `final = true` whose certificate is "unavailable" -- finality without
//      a route is the one combination the record may never publish
uint32_t CheckM1Row(const M1Row& row);
const char* M1RowMessage(uint32_t code);

// ---------------------------------------------------------------------------
// The printable table.
// ---------------------------------------------------------------------------
//
// `M1Header()` and `M1RowText()` are tab-free, single-line and stable, so a
// report can paste them and two runs can be diffed.  Every line starts with
// the literal token "M1" so that a transcript can be grepped for the table.
std::string M1Header();
std::string M1RowText(const M1Row& row);

} // namespace hecke_ext

#endif // HECKE_EXT_M1_TABLE_H
