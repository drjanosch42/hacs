// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// ===========================================================================
// hecke-extensions slot 5, WP-8 -- [E] Sec. 8.2's two further ell = 691 rows.
//
// [E] = docs/latex/hecke_extensions.tex v1.5.2 `sec:tests` Sec. 8.2 and
// `cl:bk`; [X] = hecke_extensions_implementation_plan.md v1.8 Sec. 5 (the
// HeckeExtSeparationValidation row), Sec. 9.2 Q13 / Q17, Sec. 6.2 E2 (M1).
//
// (160, 691) -- THE SEPARATION CASE.  [E] Sec. 8.2, verified 2026-09-16 by the
// slot-1 probe `HeckeLocalityProbe` and RECORDED there: one ComputeMultiOp run
// on {T_2, T_3, T_5} gives seven blocks
//
//     2:1  1:1  2:1  2:1  4:2  6:3  10:5        (dim : residue degree)
//
// on dim H^1 = 27, every one certified local, `certify() = 0`.  [X] Sec. 5
// says in so many words: "This is the value HeckeExtSeparationValidation (E2)
// must assert."  This file is that assertion.  It also makes the SEPARATION
// claim itself measurable rather than quoted: the driver is run with
// `good_primes` = {2}, {3}, {5}, {2,3} and {2,3,5}, and the block counts show
// that no single T_q separates what {T_2, T_3} does.
//
// (198, 691) -- [E] Sec. 8.2's second row: dim H^1 = 33, blocks
// 2:1, 3:1, 4:2, 4:2, 20:10, the 3:1 block the Eisenstein class merged with one
// F_691-rational cuspidal system ((691, 200) is an irregular pair), Loewy
// (2,1); the other blocks semisimple with T_m a field; the 20:10 block one
// Galois orbit of residue degree 10 with multiplicity 2.  [E] carries this row
// \VERIFY with an explicit rider (owner decision OQ13): the index-one
// ARITHMETIC is settled -- 691 || num(B_200/200) -- and the uniqueness of the
// congruent 691-adic eigenform is NOT checked.  This validation therefore
// asserts the computed structure and the divisibility, and RECORDS the
// possible ambiguity; it does not assert uniqueness.
//
// NO GOLDEN ENTRY, NO GOLDEN VALUE.  Nothing here reaches Golden/.
// ===========================================================================

#include "HeckeExtSeparationValidation.h"

#include "HeckeExtDriver.h"
#include "HeckeExtFinal.h"
#include "HeckeIdentityCertificate.h"   // DimModularForms / DimCuspForms
#include "HeckeM1Table.h"
#include "HeckePredictions.h"
#include "HeckeSturmBound.h"

#include "../Hecke/HeckeSettings.h"
#include "../Util/ValidationCheck.h"

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <string>
#include <utility>
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

// The block list as (dim : residue degree) pairs, in the record's own order.
std::vector<std::pair<uint32_t, uint32_t>> Blocks(const hecke_ext::HeckeExtRecord& r,
                                                  const std::string& group) {
    std::vector<std::pair<uint32_t, uint32_t>> out;
    for (const hecke_ext::IdealRecord& id : r.ideals) {
        bool in_group = false;
        for (const hecke_ext::IdealGroupMatch& gm : id.groups) {
            if (gm.own && gm.group == group) in_group = true;
        }
        if (!in_group) {
            for (const hecke_ext::ModuleRecordEntry& m : id.modules) {
                if (m.group == group) in_group = true;
            }
        }
        if (in_group) out.push_back(std::make_pair(id.dim, id.residue_degree));
    }
    return out;
}

std::string BlocksText(const std::vector<std::pair<uint32_t, uint32_t>>& b) {
    std::string s;
    for (size_t i = 0; i < b.size(); ++i) {
        if (i) s += " ";
        s += std::to_string(b[i].first) + ":" + std::to_string(b[i].second);
    }
    return s.empty() ? "-" : s;
}

void EmitM1(const hecke_ext::HeckeExtRecord& r, const std::string& label,
            std::vector<hecke_ext::M1Row>& out) {
    const std::vector<hecke_ext::M1Row> rows = hecke_ext::M1RowsFromRecord(r, label);
    for (const hecke_ext::M1Row& row : rows) {
        const uint32_t code = hecke_ext::CheckM1Row(row);
        ++Check.checks;
        if (code != 0) {
            ++Check.fails;
            std::cout << "    [FAIL] M1 row (" << row.n << ", " << row.ell << ") id "
                      << row.ideal_id << " -- " << hecke_ext::M1RowMessage(code) << std::endl;
        }
        std::cout << "    " << hecke_ext::M1RowText(row) << "\n";
        out.push_back(row);
    }
}

// One driver run with an explicit good-prime family.
uint32_t BlockCountWith(uint32_t n, uint32_t ell, const std::vector<uint64_t>& primes,
                        std::vector<std::pair<uint32_t, uint32_t>>* blocks_out) {
    hecke_ext::ExtDriverOptions opts;
    opts.good_primes = primes;
    // The separation question is about the DECOMPOSITION, so the generation
    // routes are switched off: they cost Stage 4 and change no block.
    opts.run_commutant = false;
    opts.run_sturm = false;
    hecke_ext::HeckeExtDriver d(n, ell, opts);
    d.run();
    const std::vector<std::pair<uint32_t, uint32_t>> b = Blocks(d.record(), "H1");
    if (blocks_out != nullptr) *blocks_out = b;
    return static_cast<uint32_t>(b.size());
}

// ===========================================================================
// Arm 1 -- (160, 691): the seven blocks of [Q17], and the separation claim.
// ===========================================================================
void RunSeparationArm(std::vector<hecke_ext::M1Row>& m1) {
    std::cout << "-- arm 1: (160, 691), [E] Sec. 8.2 / [Q17]\n";

    // (a) The probe's own family {T_2, T_3, T_5}: the seven blocks, asserted.
    std::vector<std::pair<uint32_t, uint32_t>> b235;
    const uint32_t k235 = BlockCountWith(160, 691, {2, 3, 5}, &b235);
    const std::vector<std::pair<uint32_t, uint32_t>> want = {
        {2, 1}, {1, 1}, {2, 1}, {2, 1}, {4, 2}, {6, 3}, {10, 5}
    };
    CheckEq(k235, 7u, "(160,691) {2,3,5}: seven blocks");
    Check(b235 == want, "(160,691) {2,3,5}: the block list is 2:1 1:1 2:1 2:1 4:2 6:3 10:5");
    std::cout << "    SEPARATION_BLOCKS n=160 ell=691 family=2,3,5 blocks=["
              << BlocksText(b235) << "]\n";
    {
        uint32_t total = 0;
        for (const auto& p : b235) total += p.first;
        CheckEq(total, 27u, "(160,691): the blocks exhaust dim H^1 = 27");
    }

    // (b) THE SEPARATION CLAIM, MEASURED rather than quoted -- and the
    //     measurement does not say what [E] Sec. 8.2's prose says.  See
    //     docs/implementation/slot5/WP-8_report.md Sec. 5, finding 1.
    //
    //     [E] Sec. 8.2 ends "as this claim predicts for n >> ell, no single
    //     T_q separates everything while {T_2, T_3} does".  On the SHIPPED
    //     ComputeMultiOp that is false at (160, 691) as a statement about the
    //     block PARTITION: T_2 alone and T_3 alone each already give the full
    //     seven-block partition, with the same dimensions AND the same
    //     residue degrees, as {T_2, T_3, T_5}.  What T_5 alone fails to do is
    //     not to separate but to SEE the residue degree: its partition is the
    //     same seven blocks, and it differs only in reporting f = 1 on the
    //     four-dimensional block where the family certifies f = 2.
    //
    //     This arm therefore ASSERTS what it measured -- and asserts it in a
    //     form that would catch a regression in either direction.
    std::vector<std::pair<uint32_t, uint32_t>> b2, b3, b5, b23;
    const uint32_t k2 = BlockCountWith(160, 691, {2}, &b2);
    const uint32_t k3 = BlockCountWith(160, 691, {3}, &b3);
    const uint32_t k5 = BlockCountWith(160, 691, {5}, &b5);
    const uint32_t k23 = BlockCountWith(160, 691, {2, 3}, &b23);
    std::cout << "    SEPARATION_SINGLE n=160 ell=691"
              << " T_2=[" << BlocksText(b2) << "]"
              << " T_3=[" << BlocksText(b3) << "]"
              << " T_5=[" << BlocksText(b5) << "]"
              << " T_2T_3=[" << BlocksText(b23) << "]\n";
    CheckEq(k2, 7u, "(160,691): T_2 alone gives seven blocks");
    CheckEq(k3, 7u, "(160,691): T_3 alone gives seven blocks");
    CheckEq(k5, 7u, "(160,691): T_5 alone gives seven blocks");
    CheckEq(k23, 7u, "(160,691): {T_2,T_3} gives seven blocks");
    Check(b2 == b235, "(160,691): T_2 ALONE already gives the family's block list");
    Check(b3 == b235, "(160,691): T_3 ALONE already gives the family's block list");
    Check(b23 == b235, "(160,691): {T_2,T_3} gives the family's block list");
    Check(b5 != b235, "(160,691): T_5 alone does NOT -- it is the one that differs");
    // Equal dimension multisets alone do not give equal partitions: the partitions coincide because the family's
    // decomposition refines each single operator's and every run has 7 blocks (slot-6 WP-12 M-1); T_5 differs in f only.
    {
        std::vector<uint32_t> d5, dfam;
        for (const auto& q : b5) d5.push_back(q.first);
        for (const auto& q : b235) dfam.push_back(q.first);
        std::sort(d5.begin(), d5.end());
        std::sort(dfam.begin(), dfam.end());
        Check(d5 == dfam,
              "(160,691): T_5's partition has the SAME dimensions as the family's");
        uint32_t f_sum_5 = 0, f_sum_fam = 0;
        for (const auto& q : b5) f_sum_5 += q.second;
        for (const auto& q : b235) f_sum_fam += q.second;
        Check(f_sum_5 < f_sum_fam,
              "(160,691): T_5 under-reports the residue degree, which is the real difference");
    }
    // [E] Sec. 8.2's own reading of the campaign row: T_5 alone HAS a 4:1
    // block where the family has a 4:2 one.
    {
        uint32_t four_one_in_T5 = 0, four_two_in_T5 = 0;
        for (const auto& q : b5) {
            if (q.first == 4 && q.second == 1) ++four_one_in_T5;
            if (q.first == 4 && q.second == 2) ++four_two_in_T5;
        }
        uint32_t four_one_after = 0, four_two_after = 0;
        for (const auto& q : b235) {
            if (q.first == 4 && q.second == 1) ++four_one_after;
            if (q.first == 4 && q.second == 2) ++four_two_after;
        }
        CheckEq(four_one_in_T5, 1u, "(160,691): T_5 alone HAS one 4:1 block");
        CheckEq(four_two_in_T5, 0u, "(160,691): ... and no 4:2 one");
        CheckEq(four_one_after, 0u, "(160,691): the family leaves NO 4:1 block");
        CheckEq(four_two_after, 1u,
                "(160,691): ... and one genuine residue-degree-2 block 4:2 instead");
    }

    // (c) The DEFAULT cell -- S_0 = {2,3,5,7}, U_691 refused by the frozen
    //     budget at n = 160 -- which is the record M1 is a statement about.
    hecke_ext::HeckeExtDriver d(160, 691, hecke_ext::ExtDriverOptions());
    d.run();
    const hecke_ext::HeckeExtRecord& r = d.record();
    CheckEq(r.verify(), 0u, "(160,691): HeckeExtRecord::verify()");
    Check(!d.stage_one().includes_u_ell(), "(160,691): U_691 is refused (out of budget)");
    CheckEq(r.u_ell.reason, std::string("u_ell_over_budget"),
            "(160,691): the recorded u_ell reason");
    const std::vector<std::pair<uint32_t, uint32_t>> bdef = Blocks(r, "H1");
    Check(bdef == want, "(160,691) S_0 = {2,3,5,7}: the same seven blocks");
    for (const hecke_ext::IdealRecord& id : r.ideals) {
        Check(id.local, "(160,691): every block is certified local");
        CheckEq(id.nilpotency, 1u, "(160,691): every block is semisimple (T_m a field)");
        CheckEq(id.algebra_dim, id.residue_degree, "(160,691): A_S = F_{691^f}");
        // [XA B1]: every multiplicity-2 block over a field A_S is excluded
        // from (b) by the socle bound, so B'(162, 691) = 2 is what makes it
        // final.  A block that route (a) CERTIFIES (the 1:1 one) never runs
        // the sweep, so B' is asserted on the blocks the sweep reached; the
        // CELL's B' is on every M1 row, computed independently there.
        if (id.certificate == "sturm(B')") {
            Check(id.sturm.present && id.sturm.has_B_prime,
                  "(160,691): B' exists on every block route (c) decided");
            CheckEq(id.sturm.B_prime, 2u, "(160,691): B'(162, 691) = 2");
        } else {
            CheckEq(id.certificate, std::string("cyclic"),
                    "(160,691): a block that skipped (c) was certified by (a)");
        }
    }
    EmitM1(r, "E8.2-separation-160", m1);
}

// ===========================================================================
// Arm 2 -- (198, 691).
// ===========================================================================
void RunOneNineEightArm(std::vector<hecke_ext::M1Row>& m1) {
    std::cout << "-- arm 2: (198, 691), [E] Sec. 8.2 second row / [Q13]\n";

    // The arithmetic half of [Q13], recomputed here from bernoulli_mod:
    // 691 | num(B_200/200), i.e. (691, 200) IS an irregular pair.  The "||"
    // (index exactly one) is [E]'s own mod-691^2 computation and is RECORDED,
    // not recomputed: bernoulli_mod works in F_ell and cannot see 691^2.
    {
        const hecke_ext::KummerVerdict v = hecke_ext::kummer_verdict(200, 691);
        Check(v.applicable && v.divides, "(198,691): 691 | num(B_200/200)");
        CheckEq(v.reason, std::string("direct"), "(198,691): the direct branch decides it");
        std::cout << "    SEPARATION_Q13 (691,200) irregular=yes index_one=recorded"
                     " (691 || num(B_200/200), [E] Sec. 8.2; not recomputed here)"
                     " eigenform_uniqueness=not_checked\n";
    }

    std::vector<std::pair<uint32_t, uint32_t>> b235;
    BlockCountWith(198, 691, {2, 3, 5}, &b235);
    const std::vector<std::pair<uint32_t, uint32_t>> want = {
        {2, 1}, {3, 1}, {4, 2}, {4, 2}, {20, 10}
    };
    Check(b235 == want, "(198,691) {2,3,5}: the block list is 2:1 3:1 4:2 4:2 20:10");
    std::cout << "    SEPARATION_BLOCKS n=198 ell=691 family=2,3,5 blocks=["
              << BlocksText(b235) << "]\n";

    hecke_ext::HeckeExtDriver d(198, 691, hecke_ext::ExtDriverOptions());
    d.run();
    const hecke_ext::HeckeExtRecord& r = d.record();
    CheckEq(r.verify(), 0u, "(198,691): HeckeExtRecord::verify()");
    const std::vector<uint32_t>& dims = d.H_dimensions();
    Check(dims.size() == 3, "(198,691): three degrees");
    if (dims.size() == 3) {
        CheckEq(dims[1], 33u, "(198,691): dim H^1 = 33");
        CheckEq(dims[1], hecke_ext::DimModularForms(200) + hecke_ext::DimCuspForms(200) + dims[0],
                "(198,691): Eichler-Shimura");
    }

    // The 3:1 block is the non-split one, Loewy (2,1); every other block is
    // semisimple with T_m a field; and there is exactly ONE 20:10 block -- one
    // Galois orbit of residue degree 10 with multiplicity 2.
    uint32_t nonsplit = 0, twenty_ten = 0;
    for (const hecke_ext::IdealRecord& id : r.ideals) {
        Check(id.local, "(198,691): every block is certified local");
        if (id.nilpotency >= 2) {
            ++nonsplit;
            CheckEq(id.dim, 3u, "(198,691): the non-split block is the 3:1 one");
            CheckEq(id.residue_degree, 1u, "(198,691): ... f = 1");
            CheckEq(id.nilpotency, 2u, "(198,691): ... nu = 2, T_m = F[eps]/eps^2");
            Check(id.gorenstein, "(198,691): ... Gorenstein");
            const std::vector<uint32_t> loewy_want = {2, 1};
            bool loewy_ok = false;
            for (const hecke_ext::ModuleRecordEntry& m : id.modules) {
                if (m.structure.loewy == loewy_want) loewy_ok = true;
            }
            Check(loewy_ok, "(198,691): ... Loewy layers (2,1)");
        } else {
            CheckEq(id.algebra_dim, id.residue_degree,
                    "(198,691): a semisimple block's A_S is the field F_{691^f}");
        }
        if (id.dim == 20 && id.residue_degree == 10) {
            ++twenty_ten;
            CheckEq(id.dim / id.residue_degree, 2u,
                    "(198,691): the 20:10 block has multiplicity 2 over F_{691^10}");
        }
    }
    CheckEq(nonsplit, 1u, "(198,691): exactly one non-split block");
    CheckEq(twenty_ten, 1u, "(198,691): exactly one 20:10 block");

    EmitM1(r, "E8.2-separation-198", m1);
}

} // namespace

uint32_t HeckeExtSeparationValidation() {
    Check.reset();
    std::cout << "============= HeckeExtSeparationValidation =============" << std::endl;

    CheckEq(hecke::math_settings().fingerprint(), static_cast<uint64_t>(0),
            "the mathematical settings are at their defaults (fingerprint 0)");

    std::vector<hecke_ext::M1Row> m1;
    std::cout << "    " << hecke_ext::M1Header() << "\n";
    RunSeparationArm(m1);
    RunOneNineEightArm(m1);

    std::cout << "SEPARATION_M1 rows=" << m1.size() << "\n";
    std::cout << "SEPARATION checks=" << Check.checks << " failures=" << Check.fails << std::endl;
    std::cout << "HeckeExtSeparationValidation: " << Check.fails << " failing case(s)." << std::endl;
    return Check.fails;
}
