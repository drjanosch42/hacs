// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// ===========================================================================
// hecke-extensions slot 5, WP-8 -- [E] Sec. 8.3's KUMMER LIST on X_4.
//
// [E] = docs/latex/hecke_extensions.tex v1.5.2, `sec:tests` Sec. 8.3 item
// "Small-ell Kummer cases with n < ell - 1" and `cl:harder` / Claim 1.8;
// [X] = docs/implementation/hecke_extensions_implementation_plan.md v1.8
// Sec. 5 (the Kummer row) and Sec. 6.2 E2 (M1); [C] = gl2_contract.tex v1.6.2.
//
// THE TEN CELLS ([E] Claim 1.8, k = n + 2 <= 26, n < ell - 1):
//
//   (10, 691) (14, 3617) (16, 43867) (18, 283) (18, 617)
//   (20, 131) (20, 593)  (22, 103)   (22, 2294797) (24, 657931)
//
// WHAT IS ASSERTED, and against WHAT.  Arm 1 checks THE LIST ITSELF against an
// oracle that is not the list: slot-4 WP-7's `kummer_divides` / `bernoulli_mod`
// (HeckeExt/HeckePredictions), i.e. ell | num(B_k/k) recomputed in F_ell.  It
// also sweeps a STATED range for completeness, so the ten cells are not merely
// confirmed but are the only ones there.  Arm 2 runs each cell through
// HeckeExtDriver at the DEFAULT mathematical settings and checks the record.
// Arm 3 emits the M1 rows through HeckeExt/HeckeM1Table and asserts the M1
// statement on each: `final` tri-state, a reason FROM THE FIXED LIST, and
// B'(k, ell).
//
// NO GOLDEN ENTRY, NO GOLDEN VALUE.  Nothing here reaches Golden/; the only
// collector is the in-process digest below, defined and called in this file
// alone.
//
// TIER.  `slow`.  The ten cells at the default settings cost 37 s wall, of
// which U_3617 at (14, 3617) alone is 34 s -- see docs/implementation/slot5/
// WP-8_report.md Sec. 3, which records that the slot-5 plan's "(standard)" is
// not reachable without changing `u_ell` away from its default, and that
// [X] Sec. 5 already books this validation `slow, ~40 s`.
// ===========================================================================

#include "HeckeExtKummerValidation.h"

#include "HeckeExtDriver.h"
#include "HeckeExtFinal.h"
#include "HeckeIdentityCertificate.h"   // DimModularForms / DimCuspForms
#include "HeckeM1Table.h"
#include "HeckePredictions.h"
#include "HeckeUEllBudget.h"

#include "../Hecke/HeckeSettings.h"
#include "../../ModArithmetic/Fnv.h"
#include "../Util/ValidationCheck.h"

#include <cstdint>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace {

oc_util::Checker Check("    [FAIL] ");

template <class A, class B>
void CheckEq(const A& got, const B& want, const std::string& what) {
    ++Check.checks;
    if (!(got == static_cast<A>(want))) {
        ++Check.fails;
        std::cout << "    [FAIL] " << what << ": got " << got << ", want " << want << std::endl;
    }
}

struct Cell { uint32_t n; uint32_t ell; bool u_ell_expected; };

// [E] Claim 1.8 / [X] Sec. 5's Kummer row, in the order [E] prints them, with
// the U_ell admission the FROZEN model decides ([X] Sec. 2.2 Stage 1):
// included at ell in {691, 3617, 283, 617, 131, 593, 103}, refused at
// {43867, 2294797, 657931}.
const std::vector<Cell>& KummerCells() {
    static const std::vector<Cell> cells = {
        {10, 691,     true},
        {14, 3617,    true},
        {16, 43867,   false},
        {18, 283,     true},
        {18, 617,     true},
        {20, 131,     true},
        {20, 593,     true},
        {22, 103,     true},
        {22, 2294797, false},
        {24, 657931,  false},
    };
    return cells;
}

// ---------------------------------------------------------------------------
// A golden COLLECTOR, defined here and called from this file only ([X]
// Sec. 1.6's ext subgrid names such objects; slot 5 emits nothing from this
// package -- the slot's one emission is WP-3's).  It is an in-process
// stability digest, not a reference lookup.
// ---------------------------------------------------------------------------
uint64_t CollectKummerRows(const std::vector<hecke_ext::M1Row>& rows) {
    uint64_t h = engine::kFnvOffset64;
    for (const hecke_ext::M1Row& r : rows) {
        h = engine::FnvU64(h, r.n);
        h = engine::FnvU64(h, r.ell);
        h = engine::FnvBytes(h, r.group);
        h = engine::FnvU64(h, r.dim);
        h = engine::FnvU64(h, r.residue_degree);
        h = engine::FnvU64(h, r.algebra_dim);
        h = engine::FnvU64(h, r.nilpotency);
        h = engine::FnvBytes(h, r.certificate);
        h = engine::FnvBytes(h, r.final_value);
        h = engine::FnvBytes(h, r.final_reason);
        h = engine::FnvU64(h, r.has_B_prime ? r.B_prime : 0u);
    }
    return h;
}

std::string HexU64(uint64_t v) {
    std::ostringstream os;
    os << "0x" << std::hex << std::setfill('0') << std::setw(16) << v;
    return os.str();
}

// ===========================================================================
// Arm 1 -- the list itself, against `kummer_divides` and a completeness sweep.
// ===========================================================================
void RunListArm() {
    std::cout << "-- arm 1: the Kummer list against bernoulli_mod / kummer_divides\n";

    for (const Cell& c : KummerCells()) {
        const uint32_t k = c.n + 2;
        const std::string tag =
            "(" + std::to_string(c.n) + ", " + std::to_string(c.ell) + ")";
        // The defining condition of [E] cl:harder: ell | num(B_k / k).
        const hecke_ext::KummerVerdict v = hecke_ext::kummer_verdict(k, c.ell);
        Check(v.applicable, tag + ": the Kummer verdict is applicable");
        Check(v.divides, tag + ": ell | num(B_k/k) -- the cell IS a Kummer cell");
        // Every listed cell is inside the direct range n < ell - 1, so the
        // verdict must come from the DIRECT branch and not from a congruence.
        Check(c.n < c.ell - 1, tag + ": n < ell - 1 (the range of cl:harder)");
        CheckEq(v.reason, std::string("direct"), tag + ": the direct branch decides it");
        CheckEq(v.k_reduced, k, tag + ": no reduction was needed");
        // B_k/k is zero in F_ell exactly when ell divides its numerator.
        CheckEq(v.b_over_k_mod_ell, 0u, tag + ": B_k/k == 0 in F_ell");
    }

    // COMPLETENESS, in a STATED range.  For every odd prime ell <= 20000 and
    // every even k with 4 <= k <= 26 and k < ell - 1, ell | num(B_k/k) holds
    // for exactly the SEVEN listed pairs whose ell is in that range -- and for
    // nothing else.  That is the whole of [E] Claim 1.8's k <= 26 list except
    // the three pairs with ell > 20000 (43867, 657931, 2294797), which are
    // checked individually above.  The range is stated rather than stretched
    // to 2294797 because the sweep is O(#primes * k^2) and the claim is about
    // what it covers.
    {
        const uint32_t sweep_max = 20000;
        std::vector<char> composite(sweep_max + 1, 0);
        for (uint32_t i = 2; i * i <= sweep_max; ++i) {
            if (composite[i]) continue;
            for (uint32_t j = i * i; j <= sweep_max; j += i) composite[j] = 1;
        }
        uint32_t hits = 0, primes_swept = 0;
        std::vector<std::pair<uint32_t, uint32_t>> found;   // (k, ell)
        for (uint32_t ell = 3; ell <= sweep_max; ++ell) {
            if (composite[ell]) continue;
            ++primes_swept;
            for (uint32_t k = 4; k <= 26; k += 2) {
                if (k >= ell - 1) continue;             // outside the direct range
                if (hecke_ext::kummer_divides(k, ell)) {
                    ++hits;
                    found.push_back(std::make_pair(k, ell));
                }
            }
        }
        // The seven listed pairs with ell <= 20000, in the sweep's own order
        // (ascending ell).
        const std::vector<std::pair<uint32_t, uint32_t>> want = {
            {24, 103}, {22, 131}, {20, 283}, {22, 593}, {20, 617}, {12, 691}, {16, 3617}
        };
        CheckEq(hits, 7u, "completeness sweep (ell <= 20000, even k <= 26): exactly seven hits");
        Check(found == want,
              "completeness sweep: the hits are EXACTLY [E] Claim 1.8's pairs with ell <= 20000");
        // Every hit is a listed cell, and every listed cell with ell <= 20000
        // is a hit -- checked both ways so that neither list can drift.
        for (const auto& f : found) {
            bool listed = false;
            for (const Cell& c : KummerCells()) {
                if (c.n + 2 == f.first && c.ell == f.second) listed = true;
            }
            Check(listed, "completeness sweep: hit (" + std::to_string(f.first) + ", " +
                          std::to_string(f.second) + ") is on [E] Claim 1.8's list");
        }
        for (const Cell& c : KummerCells()) {
            if (c.ell > sweep_max) continue;
            bool hit = false;
            for (const auto& f : found) {
                if (f.first == c.n + 2 && f.second == c.ell) hit = true;
            }
            Check(hit, "completeness sweep: the listed cell (" + std::to_string(c.n) + ", " +
                       std::to_string(c.ell) + ") is found");
        }
        std::cout << "    KUMMER_SWEEP primes=" << primes_swept << " k_even<=26 hits=" << hits;
        for (const auto& f : found) std::cout << " (" << f.first << "," << f.second << ")";
        std::cout << "\n";
    }
}

// ===========================================================================
// Arm 2 + 3 -- the record on each cell, and its M1 rows.
// ===========================================================================
void RunRecordArm() {
    std::cout << "-- arm 2: the ten Kummer cells through HeckeExtDriver (default settings)\n";
    std::cout << "    " << hecke_ext::M1Header() << "\n";

    // The default configuration is what M1 is a statement about, so the
    // fingerprint must be 0 throughout ([X] Sec. 1.5: a golden or M1 value can
    // never be explained by a non-default setting).
    CheckEq(hecke::math_settings().fingerprint(), static_cast<uint64_t>(0),
            "the mathematical settings are at their defaults (fingerprint 0)");

    std::vector<hecke_ext::M1Row> all_rows;
    uint32_t cells_with_one_nonsplit = 0;

    for (const Cell& c : KummerCells()) {
        const std::string tag =
            "(" + std::to_string(c.n) + ", " + std::to_string(c.ell) + ")";
        const uint32_t k = c.n + 2;

        hecke_ext::ExtDriverOptions opts;
        hecke_ext::HeckeExtDriver d(c.n, c.ell, opts);
        d.run();
        const hecke_ext::HeckeExtRecord& r = d.record();

        CheckEq(r.verify(), 0u, tag + ": HeckeExtRecord::verify()");
        CheckEq(r.schema, std::string(hecke_ext::kExtSchema), tag + ": the v2 schema");
        CheckEq(r.n_sym, c.n, tag + ": n");
        CheckEq(r.weight, k, tag + ": k = n + 2");

        // --- the cell: X_1 = 0 and dim X_4 = dim M_k + dim S_k (Eichler-
        //     Shimura with dim H^0 = 0; n < ell - 1 is off every Dickson
        //     locus, which Hecke/DicksonCount.h is the project's one count of).
        const std::vector<uint32_t>& dims = d.H_dimensions();
        Check(dims.size() == 3, tag + ": three degrees");
        if (dims.size() == 3) {
            CheckEq(dims[0], 0u, tag + ": dim X_1 = dim H^0 = 0");
            CheckEq(dims[2], 0u, tag + ": dim H^2 = 0");
            const uint32_t want = hecke_ext::DimModularForms(k) + hecke_ext::DimCuspForms(k);
            CheckEq(dims[1], want, tag + ": dim X_4 = dim M_k + dim S_k");
        }

        // --- the U_ell admission of the FROZEN model ([X] Sec. 2.2 Stage 1).
        CheckEq(d.stage_one().includes_u_ell(), c.u_ell_expected,
                tag + ": the U_ell admission");
        Check(r.u_ell.present, tag + ": the record carries a cell-level u_ell object");
        CheckEq(r.u_ell.included, c.u_ell_expected, tag + ": the recorded u_ell verdict");
        CheckEq(r.u_ell.reason,
                std::string(c.u_ell_expected ? "within_budget" : "u_ell_over_budget"),
                tag + ": the recorded u_ell reason");
        // v2: exactly ONE u_ell decision per cell, however many ideals it has.
        {
            uint32_t ideal_u_ell = 0;
            for (const hecke_ext::IdealRecord& id : r.ideals) {
                for (const hecke_ext::ResumableRecord* res : id.all_resumables()) {
                    if (res != nullptr && res->kind == "u_ell") ++ideal_u_ell;
                }
            }
            CheckEq(ideal_u_ell, 0u, tag + ": no ideal carries a u_ell resumable (v2)");
        }

        // --- the blocks: EXACTLY ONE non-split ideal, Loewy (2,1); all the
        //     others semisimple ([E] Sec. 8.3's Kummer item).
        uint32_t nonsplit = 0;
        for (const hecke_ext::IdealRecord& id : r.ideals) {
            Check(id.local, tag + ": every block is certified local");
            Check(id.algebra_present, tag + ": every block has its local algebra");
            const bool semisimple = (id.nilpotency <= 1);
            if (!semisimple) {
                ++nonsplit;
                CheckEq(id.nilpotency, 2u, tag + ": the non-split block has nu = 2");
                CheckEq(id.embdim, 1u, tag + ": ... embedding dimension 1");
                Check(id.gorenstein, tag + ": ... A_S is Gorenstein");
                CheckEq(id.algebra_dim, 2u * id.residue_degree,
                        tag + ": ... dim A_S = 2f, i.e. F_{ell^f}[eps]/eps^2");
                CheckEq(id.residue_degree, 1u, tag + ": ... f = 1");
                CheckEq(id.dim, 3u, tag + ": ... D = 3 (T_m (+) F)");
                const std::vector<uint32_t> want = {2, 1};
                bool loewy_ok = false, socle_ok = false;
                for (const hecke_ext::ModuleRecordEntry& m : id.modules) {
                    if (m.structure.loewy == want) loewy_ok = true;
                    if (m.structure.socle == want) socle_ok = true;
                }
                Check(loewy_ok, tag + ": ... Loewy layers (2,1)");
                Check(socle_ok, tag + ": ... socle layers (2,1)");
            } else {
                CheckEq(id.algebra_dim, id.residue_degree,
                        tag + ": a semisimple block's A_S is the field F_{ell^f}");
            }
        }
        CheckEq(nonsplit, 1u, tag + ": exactly one non-split Eisenstein-cuspidal ideal");
        if (nonsplit == 1) ++cells_with_one_nonsplit;

        // --- generation ([XA B1] / [X] Sec. 5's Kummer row): B' = 2 on every
        //     cell, and the non-split block is final ONLY via sturm(2).
        for (const hecke_ext::IdealRecord& id : r.ideals) {
            Check(id.sturm.present, tag + ": the sturm object is emitted");
            Check(id.sturm.has_B_prime, tag + ": B'(k, ell) exists");
            CheckEq(id.sturm.B_prime, 2u, tag + ": B'(k, ell) = 2");
            if (id.nilpotency >= 2) {
                CheckEq(id.certificate_routes.a, std::string("not_cyclic"),
                        tag + ": (a) fails on the non-split block");
                CheckEq(id.certificate_routes.b, std::string("excluded_by_socle_bound"),
                        tag + ": (b) is excluded by the socle bound -- a PROOF of failure");
                CheckEq(id.certificate, std::string("sturm(B')"),
                        tag + ": the non-split block is final only via (c)");
            }
        }

        // --- the M1 rows.
        const std::vector<hecke_ext::M1Row> rows =
            hecke_ext::M1RowsFromRecord(r, "E8.3-kummer");
        CheckEq(static_cast<uint32_t>(rows.size()),
                static_cast<uint32_t>(r.ideals.size()), tag + ": one M1 row per ideal");
        for (const hecke_ext::M1Row& row : rows) {
            const uint32_t code = hecke_ext::CheckM1Row(row);
            if (code != 0) {
                Check(false, tag + ": M1 row " + std::to_string(row.ideal_id) + " -- " +
                                 hecke_ext::M1RowMessage(code));
            } else {
                ++Check.checks;
            }
            CheckEq(row.B_prime, 2u, tag + ": the M1 row's B' column is 2");
            std::cout << "    " << hecke_ext::M1RowText(row) << "\n";
            all_rows.push_back(row);
        }
    }

    CheckEq(cells_with_one_nonsplit, static_cast<uint32_t>(KummerCells().size()),
            "every Kummer cell has exactly one non-split ideal");

    std::cout << "    KUMMER_M1 cells=" << KummerCells().size()
              << " rows=" << all_rows.size()
              << " digest=" << HexU64(CollectKummerRows(all_rows)) << "\n";
}

} // namespace

uint32_t HeckeExtKummerValidation() {
    Check.reset();
    std::cout << "=============== HeckeExtKummerValidation ===============" << std::endl;
    RunListArm();
    RunRecordArm();
    std::cout << "KUMMER checks=" << Check.checks << " failures=" << Check.fails << std::endl;
    std::cout << "HeckeExtKummerValidation: " << Check.fails << " failing case(s)." << std::endl;
    return Check.fails;
}
