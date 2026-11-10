// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// ===========================================================================
// hecke-extensions slot 13, WP-9 -- HeckeCensusLatticeValidation (standard).
//
// C-F9 ([PR] = docs/implementation/slot12/WP-12_finality_proposal.md v2 Sec. 7)
// tested on census edition 2 by HeckeExt/HeckeCensusLattice (see its header).
// C-F9 is a CANDIDATE: this validation gates the instrument, not the candidate.
//
//   (P) the detector on planted modules over Z/7^2: a Loewy-length-3 Jordan
//       block fails the pair test, a Loewy-length-2 one passes, ell * (a
//       length-3 block) passes (its square is in ell M), the torsion rows
//       (a_i = 1) read mod ell.
//   (S) a fixed row sample of the census's non-reduced rows -- (34, 7),
//       (82, 7) [the RCOMP datum], (142, 7) [length 4, embdim 3], (54, 11),
//       (38, 13) [cuspidal], (10, 691) [the 691 | B_12 congruence]: per row
//       the instrument's consistency (the F_ell and Z/ell^2 routes agree,
//       rank e_m = dim_E, e_m idempotent, H^1(Z/ell^2) has dim H^1(F_ell)
//       generators, m_A != 0 detected, right exactness of the Serre-weight
//       filtration), census agreement (dim_E, dim_plus, dim_minus, Loewy
//       length of V = loewy_A) and the C-F9 status PINNED to the value the
//       full run observed (slot-13 WP-9: held on every sampled row) -- a change
//       is a finding to be explained, not a test to be repaired.
//   (ALL) HECKE_CENSUS_LATTICE_ALL=1 (read-only, not an emitter): every
//       non-reduced row of HECKE_CENSUS_LATTICE_TSV (default the tracked
//       edition 2), one CENSUS_LATTICE line per row and one
//       CENSUS_LATTICE_SUMMARY line; gated on the instrument's consistency
//       and census agreement only -- the per-row C-F9 status is REPORTED.
//       HECKE_CENSUS_LATTICE_CELLS="n:ell,..." restricts it (sharding).
//
//       (slot-14 WP-2, [C3]/[C4] cross-architecture finding: cpu2's 11
//       `census_lattice_s0..s10` shards were all vacuous -- they set only
//       CELLS, never ALL, so every shard silently ran just the (P)/(S) default
//       arms and PASSed in ~72 s with none of the sharded rows checked.)
//       HECKE_CENSUS_LATTICE_CELLS ALONE now enables arm (ALL) as well,
//       exactly as HECKE_CENSUS_SPLIT_CELLS / HECKE_THETA_IMAGE_CELLS already
//       do for their own validations -- the env-hook table is corrected to
//       match. ALL=1 with no CELLS still runs the full TSV; CELLS wins over
//       ALL=1 when both are set (unchanged).
//
//       (slot-16 WP-3, WP-12 Sec. 8.1: the (654, 37) ideals 17 / 18 merged,
//       every later id shifted, and the rows were joined by id.)  The re-run
//       now takes u_ell from the row's SOURCE SET (HeckeCensusJoin.h; np / npx
//       16384, ext_u691 8192, else the default; HECKE_CENSUS_LATTICE_U_ELL=<mb>
//       overrides; an unknown set is refused by TSV line number) and joins every
//       row to the re-run ideal BY EIGEN-SYSTEM (the record's split over its
//       primes, U_ell included; records under HECKE_CENSUS_LATTICE_RECORDS, a
//       ':' list, default the tree's results/), falling back to (id,
//       witness_poly, module_D) where no F_ell system or no record exists.  A
//       failed join is a CENSUS_LATTICE_REFUSED line and a failing check.  The
//       EFFECTIVE line prints operators=, u_ell= and join=.  Census columns that
//       read not_computed (the np sets' dim_E / dim_plus / dim_minus) are not
//       compared (dim_E against module_D instead).
//       HECKE_CENSUS_LATTICE_JOIN_ONLY=1 runs the driver and the join only (no
//       Z/ell^2 module run, no C-F9): arm=JOIN, a cheap pre-flight.
//
//       (slot-16 WP-3 audit repairs, docs/implementation/slot16/WP-3_WP-5_audit.md.)  The witness
//       join checks the operator set (U_ell in the re-run iff in the record; without a record, a
//       budgeted row needs U_ell) and must match uniquely; a residue-degree-1 row that fell back is a
//       CENSUS_LATTICE_FALLBACK line and counted (`fallback:` on EFFECTIVE, with `records=<roots>`);
//       an explicit HECKE_CENSUS_LATTICE_RECORDS that names no root or a missing directory fails, and
//       so does an explicit list under which every residue-degree-1 row fell back.  A malformed CELLS
//       token is refused by position and a CELLS list selecting no row fails.  The U_ELL override
//       waives the unknown-set refusal (as in arm (T)).
//   (J) the join machinery, always on (a few ms): the record reader on a tracked record, the u_ell
//       table against WP-3_expected_uell.tsv, malformed records, JoinRow on synthetic cells (the
//       WP-12 Sec. 8.1 merged-ideal negative control included).
// ===========================================================================

#include "HeckeCensusLatticeValidation.h"

#include "HeckeCensusJoin.h"
#include "HeckeCensusLattice.h"

#include "../../ModArithmetic/EngineCheck.h"
#include "../Util/ValidationCheck.h"

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <map>
#include <set>
#include <stdexcept>
#include <sstream>
#include <string>
#include <vector>

namespace {

using hecke::ordinary::RingMat;
using namespace hecke_ext;

oc_util::Checker Check("  FAIL: ", false);

using Row = std::map<std::string, std::string>;

std::vector<Row> ReadTsv(const std::string& path) {
    std::ifstream in(path);
    if (!in.good()) throw std::runtime_error("HeckeCensusLatticeValidation: cannot open the census TSV " + path);
    std::vector<Row> rows;
    std::vector<std::string> header;
    std::string line;
    auto split = [](const std::string& s) {
        std::vector<std::string> out;
        std::string cur;
        for (char c : s) {
            if (c == '\t') { out.push_back(cur); cur.clear(); }
            else if (c != '\r') cur.push_back(c);
        }
        out.push_back(cur);
        return out;
    };
    uint64_t lineno = 0;   // (slot-16 WP-3) the physical TSV line, for refusals by line number
    while (std::getline(in, line)) {
        ++lineno;
        if (line.empty() || line[0] == '#') continue;
        if (header.empty()) { header = split(line); continue; }
        const std::vector<std::string> f = split(line);
        ENGINE_CHECK_MSG(f.size() == header.size(), "HeckeCensusLatticeValidation: a census row of the wrong width");
        Row r;
        for (size_t i = 0; i < f.size(); ++i) r[header[i]] = f[i];
        r["#line"] = std::to_string(lineno);
        rows.push_back(std::move(r));
    }
    return rows;
}

std::string CensusPath() {
    const char* e = std::getenv("HECKE_CENSUS_LATTICE_TSV");
    if (e && *e) return e;
    return std::string(OC_SOURCE_DIR) + "/docs/matrix-engine/results/congruence_census/edition2/census.tsv";
}

bool EnvOn(const char* name) {
    const char* e = std::getenv(name);
    return e && std::string(e) == "1";
}

RingMat Make(uint64_t ell, const std::vector<uint32_t>& ex, const std::vector<std::vector<uint64_t>>& a) {
    RingMat M;
    M.ell = ell;
    M.row_exp = ex;
    for (const auto& r : a)
        for (uint64_t x : r) M.a.push_back(x);
    return hecke::ordinary::Canonical(M);
}

// ---------------------------------------------------------------------------
void ArmP() {
    std::cout << "-- (P) the detector on planted modules over Z/7^2\n";
    const uint64_t ell = 7;
    const std::vector<uint32_t> f3 = {2, 2, 2};
    const RingMat I3 = hecke::ordinary::Identity(ell, f3);
    const RingMat J3 = Make(ell, f3, {{0, 1, 0}, {0, 0, 1}, {0, 0, 0}});
    const RingMat J2 = Make(ell, f3, {{0, 1, 0}, {0, 0, 0}, {0, 0, 0}});
    const RingMat L3 = Make(ell, f3, {{0, 7, 0}, {0, 0, 7}, {0, 0, 0}});
    std::string ff;
    uint32_t np = 0;
    Check(!PairProductsInEllM({J3}, I3, &ff, &np) && ff == "0,0" && np == 1,
          "(P1) a Loewy-length-3 Jordan block fails the pair test at (0,0)");
    Check(PairProductsInEllM({J2}, I3, nullptr, nullptr), "(P2) a Loewy-length-2 block passes");
    Check(PairProductsInEllM({L3}, I3, nullptr, nullptr), "(P3) ell * (a length-3 block) passes (square in ell M)");
    // two commuting operators whose product is non-zero mod ell but each squares to zero
    const std::vector<uint32_t> f4 = {2, 2, 2, 2};
    const RingMat I4 = hecke::ordinary::Identity(ell, f4);
    // x = E_{12} + E_{34}... use the algebra F[x,y]/(x^2, y^2) on basis 1, x, y, xy
    const RingMat X = Make(ell, f4, {{0, 0, 0, 0}, {1, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 1, 0}});
    const RingMat Y = Make(ell, f4, {{0, 0, 0, 0}, {0, 0, 0, 0}, {1, 0, 0, 0}, {0, 1, 0, 0}});
    ff.clear();
    Check(!PairProductsInEllM({X, Y}, I4, &ff, &np) && ff == "0,1" && np == 3,
          "(P4) F[x,y]/(x^2,y^2): x^2 = y^2 = 0 but xy != 0 -- the mixed pair (0,1) fails, 3 pairs");
    // torsion rows: on Z/7 (+) Z/7^2 the entry 7 in a Z/7^2 row is in ell M, an entry 1 in a Z/7 row is not
    const std::vector<uint32_t> ft = {1, 2};
    const RingMat It = hecke::ordinary::Identity(ell, ft);
    const RingMat T1 = Make(ell, ft, {{0, 0}, {7, 0}});
    const RingMat T2 = Make(ell, ft, {{0, 1}, {0, 0}});
    Check(PairProductsInEllM({T1, T1}, It, nullptr, nullptr), "(P5) products landing in ell (Z/ell^2) pass");
    Check(hecke::ordinary::RankModEll(hecke::ordinary::Mul(T2, T1)) == 0,
          "(P5') 7 in a Z/7 row is 0: the torsion rows read mod ell");
    const RingMat T3 = Make(ell, ft, {{0, 0}, {1, 0}});
    Check(!PairProductsInEllM({T3, T2}, It, nullptr, nullptr), "(P5'') a unit entry in a product fails");
    // a projector cutting out a summand: the pair test is read on e M only
    const RingMat E = Make(ell, f3, {{0, 0, 0}, {0, 1, 0}, {0, 0, 1}});
    const RingMat J3e = Make(ell, f3, {{0, 0, 0}, {1, 0, 0}, {0, 0, 0}});
    Check(PairProductsInEllM({J3e}, E, nullptr, nullptr), "(P6) an operator killing e M passes on e M");
}

struct Pin {
    uint32_t n, ell;
    const char* status;   // the full run's observation for every sampled non-reduced row
};

// (slot-16 WP-3) split in two: the census agreement (also the whole check of a join-only row) and the
// instrument's consistency; CheckRowAgainstCensus runs both, in the order it always did.
void CheckCensusAgreement(const CensusLatticeRow& r, const Row& t, const std::string& tag) {
    // (slot-16 WP-3) A census column the row's data set did not compute (`not_computed`: the np / npx
    // sets' dim_E, dim_plus, dim_minus) is not compared -- before, those rows failed here by
    // construction (slot-15 WP-15 Sec. 2).  module_D is the block dimension on every row; the
    // edition-2 rows with dim_E computed have dim_E == module_D (2568 / 2568 H1 rows).
    auto computed = [&](const char* col) { return t.count(col) && t.at(col) != "not_computed"; };
    if (computed("dim_E")) Check(std::to_string(r.dim_E) == t.at("dim_E"), tag + ": dim_E agrees with the census");
    else if (computed("module_D"))
        Check(std::to_string(r.dim_E) == t.at("module_D"), tag + ": dim_E agrees with the census's module_D");
    if (r.eigen_ok && computed("dim_plus") && computed("dim_minus"))
        Check(std::to_string(r.dim_plus) == t.at("dim_plus") && std::to_string(r.dim_minus) == t.at("dim_minus"),
              tag + ": dim_plus / dim_minus agree with the census");
    if (r.eigen_ok && computed("loewy_A"))   // f > 1: no signs, no Loewy lengths (not_evaluated)
        Check(std::to_string(r.loewy_V) == t.at("loewy_A"),
              tag + ": the Loewy length of V under A is the census's loewy_A (A acts faithfully on V)");
}

void CheckInstrument(const CensusLatticeCell& c, const CensusLatticeRow& r, const std::string& tag) {
    Check(c.failures.empty() && r.failures.empty(), tag + ": instrument consistency (no failure recorded)");
    Check(r.cf9_field == r.cf9_lattice, tag + ": the F_ell and Z/ell^2 routes agree");
    Check(r.lattice_rank == r.dim_E, tag + ": rank_F_ell e_m = dim_E");
    Check(r.m_nonzero, tag + ": the detector sees m_A != 0 on a non-reduced row");
    Check(!r.sw_evaluated || r.mech_dim_ok, tag + ": dim H^1_{m'} <= the Glover-factor sum (right exactness)");
}

void CheckRowAgainstCensus(const CensusLatticeCell& c, const CensusLatticeRow& r, const Row& t,
                           const std::string& tag) {
    CheckCensusAgreement(r, t, tag);
    CheckInstrument(c, r, tag);
}

void ArmS() {
    std::cout << "-- (S) the fixed row sample\n";
    const std::vector<Row> census = ReadTsv(CensusPath());
    const std::vector<Pin> pins = {{34, 7, "held"},  {82, 7, "held"},  {142, 7, "held"},
                                   {54, 11, "held"}, {38, 13, "held"}, {10, 691, "held"}};
    CensusWeightCache weights;
    uint32_t rows_seen = 0;
    for (const Pin& p : pins) {
        std::map<uint32_t, const Row*> nr;
        for (const Row& t : census)
            if (t.at("n") == std::to_string(p.n) && t.at("ell") == std::to_string(p.ell) &&
                t.at("reduced") == "false" && t.at("own_group") == "H1")
                nr[static_cast<uint32_t>(std::stoul(t.at("ideal_id")))] = &t;
        Check(!nr.empty(), "(S) cell (" + std::to_string(p.n) + "," + std::to_string(p.ell)
                               + ") has a non-reduced H1 row in the census");
        CensusLatticeOptions o;
        const CensusLatticeCell c = ComputeCensusLattice(p.n, p.ell, o, &weights);
        for (const CensusLatticeRow& r : c.rows) {
            auto it = nr.find(r.ideal_id);
            if (it == nr.end()) continue;
            ++rows_seen;
            std::cout << CensusLatticeLine(c, r) << "\n";
            const std::string tag = "(S) (" + std::to_string(p.n) + "," + std::to_string(p.ell) + ") ideal "
                                    + std::to_string(r.ideal_id);
            CheckRowAgainstCensus(c, r, *it->second, tag);
            Check(r.status == p.status, tag + ": C-F9 status " + r.status + " == pinned " + p.status);
        }
    }
    Check(rows_seen >= 6, "(S) at least six sampled rows");
}


// (slot-16 WP-3 audit repair, finding 4) HECKE_CENSUS_LATTICE_CELLS through census::ParseCellList: a
// malformed token is refused by its position (a failing check), never dropped, and a set variable
// enables the arm even when every token is bad -- so "654-37" fails loudly instead of running nothing.
census::CellList CellFilter() { return census::ParseCellList("HECKE_CENSUS_LATTICE_CELLS"); }

// (slot-16 WP-3) The census join.  The re-run's ideal ids equal the record's only under the record's
// operator set, so a row is joined to a re-run ideal BY ITS EIGEN-SYSTEM: the record's split
// (`T_q mod m` over split.primes, U_ell included when the record ran it), read from the row's
// record_path under census::RecordRoots().  Exactly one re-run ideal must carry that system, every
// split prime must be a re-run operator prime (else `operator_set`: the U_ell drop of WP-12 Sec. 8.1),
// and the block dimension must be the row's module_D.  Where no F_ell system exists (residue degree
// > 1) or the record is not readable, the row is joined by (ideal_id, witness_poly, module_D) and the
// mode `witness` is printed.  A failed join is a REFUSAL: a CENSUS_LATTICE_REFUSED line and a failing
// check, never a pass.
//
// (slot-16 WP-3 audit repair, finding 2) The witness branch is no longer a bare id join:
//   * the operator set is checked first (census::CheckUEllOperator): with a readable record, U_ell is in
//     the re-run iff it is in the record; without one, a row whose u_ell budget was set (np / npx,
//     ext_u691, an override) needs U_ell in the re-run -- else `operator_set:...`, refused;
//   * the witness match must be UNIQUE (`witness_ambiguous` otherwise);
//   * a residue-degree-1 row that had to fall back (no record, or an unusable one) is a FALLBACK: it is
//     reported per row (CENSUS_LATTICE_FALLBACK, with the reason) and counted on the EFFECTIVE line, and
//     when the record roots were given explicitly (HECKE_CENSUS_LATTICE_RECORDS) a run in which every
//     residue-degree-1 row fell back fails.
struct JoinResult {
    const CensusLatticeRow* r = nullptr;
    std::string mode;       // system | witness
    std::string why;        // the refusal reason when r == nullptr
    std::string fallback;   // non-empty: a residue-degree-1 row joined by witness because its record was unusable
};

JoinResult JoinRow(const CensusLatticeCell& c, const Row& t, const census::RecordSystems& rs,
                   const std::vector<uint64_t>& S, uint32_t budget_mb) {
    JoinResult j;
    const uint32_t id = static_cast<uint32_t>(std::stoul(t.at("ideal_id")));
    const std::vector<uint32_t> wp = census::ParseUIntList(t.at("witness_poly"));
    const bool has_D = t.count("module_D") && t.at("module_D") != "not_computed";
    const std::string D = has_D ? t.at("module_D") : std::string();
    const bool f1 = !t.count("residue_degree") || t.at("residue_degree") == "1";
    const bool with_u = std::find(S.begin(), S.end(), static_cast<uint64_t>(c.ell)) != S.end();
    if (rs.ok) {
        if (rs.n != c.n || rs.ell != c.ell) {
            j.why = "record_cell_mismatch";
            return j;
        }
        const auto it = rs.ideals.find(id);
        if (it == rs.ideals.end()) {
            j.why = "record_has_no_ideal_" + std::to_string(id);
            return j;
        }
        const census::RecordIdeal& ri = it->second;
        if (ri.witness_poly != wp) {
            j.why = "record_witness_poly_differs";
            return j;
        }
        if (ri.split_ok && ri.r == 1) {
            j.mode = "system";
            std::vector<const CensusLatticeRow*> hits;
            for (const CensusLatticeRow& x : c.rows) {
                if (!x.eigen_ok) continue;
                const std::string w = census::CompareSystem(ri, x.primes, x.a);
                if (w.rfind("operator_set", 0) == 0) {
                    j.why = w;
                    return j;
                }
                if (w.empty()) hits.push_back(&x);
            }
            if (hits.size() != 1) {
                j.why = hits.empty() ? "no_rerun_ideal_with_the_system" : "system_ambiguous";
                return j;
            }
            if (has_D && std::to_string(hits[0]->dim_E) != D) {
                j.why = "system_matched_rerun_ideal_" + std::to_string(hits[0]->ideal_id) + "_of_dim_"
                        + std::to_string(hits[0]->dim_E) + "_not_module_D_" + D;
                return j;
            }
            j.r = hits[0];
            return j;
        }
        if (f1) j.fallback = "record_split_unusable";
    } else if (f1) {
        j.fallback = rs.why.empty() ? std::string("record_unusable") : rs.why;
    }
    j.mode = "witness";
    const std::string u = census::CheckUEllOperator(rs, c.ell, with_u, budget_mb);
    if (!u.empty()) {
        j.why = u;
        return j;
    }
    uint32_t hits = 0;
    for (const CensusLatticeRow& x : c.rows)
        if (x.ideal_id == id && x.witness_poly == wp && (!has_D || std::to_string(x.dim_E) == D)) {
            j.r = &x;
            ++hits;
        }
    if (hits > 1) {
        j.r = nullptr;
        j.why = "witness_ambiguous";
        return j;
    }
    if (!j.r) j.why = std::string("witness_join_failed") + (rs.ok ? "" : ":" + rs.why);
    return j;
}

// ---------------------------------------------------------------------------
// (slot-16 WP-3 audit repair, finding 3) (J) the join machinery, always on and cheap (a few ms): the
// pieces arms (S) and (P) never reach, so a regression in the JSON reader, the u_ell table or the join
// is visible to --all.
//   (J1) ReadRecordSystems on one tracked edition-2 record (ext_first (34, 7)): n, ell, operators,
//        U_ell included, ideal 2's witness_poly and split system.
//   (J2) CensusUEllBudgetMb against docs/implementation/slot16/WP-3_expected_uell.tsv (every row's
//        wp3_budget_mb) and the table's fixed points; ParseUEllOverride; ParseCellList.
//   (J3) ParseRecordSystems refuses malformed records (a non-integer operator, a duplicate ideal id).
//   (J4) JoinRow on synthetic cells: a system join; the merged-ideal negative control of WP-12 Sec. 8.1
//        (record with U_37, re-run without it) refused `operator_set`; an ambiguous system refused; the
//        witness fallback refused under a u_ell budget when U_ell is absent and counted as a fallback
//        when it joins; a non-unique witness match refused.
census::RecordSystems Synthetic(const std::string& json) { return census::ParseRecordSystems(json, "(synthetic)"); }

CensusLatticeRow RerunIdeal(uint32_t id, uint32_t dim, const std::vector<uint64_t>& primes, const std::vector<uint32_t>& a,
                            const std::vector<uint32_t>& wp) {
    CensusLatticeRow r;
    r.ideal_id = id;
    r.dim_E = dim;
    r.primes = primes;
    r.a = a;
    r.witness_poly = wp;
    r.eigen_ok = true;
    return r;
}

void ArmJ() {
    std::cout << "-- (J) the census join machinery (record reader, u_ell table, JoinRow)\n";
    {   // (J1)
        const census::RecordSystems rs = census::ReadRecordSystems(
            "ext_first/ext/hecke_ext_ell7_n34_xd78e1136a35beef2.json",
            {std::string(OC_SOURCE_DIR) + "/docs/matrix-engine/results"});
        Check(rs.ok && rs.n == 34 && rs.ell == 7, "(J1) the tracked (34, 7) record reads: n = 34, ell = 7 (" + rs.why + ")");
        Check(rs.operators == std::vector<uint64_t>({2, 3, 5, 7}) && rs.u_ell_included,
              "(J1) operators_computed = [2,3,5,7], u_ell.included = true");
        const auto it = rs.ideals.find(2);
        Check(rs.ideals.size() == 3 && it != rs.ideals.end(), "(J1) three ideals, ideal 2 present");
        if (it != rs.ideals.end())
            Check(it->second.split_ok && it->second.r == 1 && it->second.witness_poly == std::vector<uint32_t>({3, 1}) &&
                      it->second.primes == std::vector<uint64_t>({2, 3, 5, 7}) &&
                      it->second.a == std::vector<uint32_t>({4, 0, 0, 0}),
                  "(J1) ideal 2: witness_poly [3,1], split r = 1 over [2,3,5,7] = (4,0,0,0)");
        const census::RecordSystems none = census::ReadRecordSystems("no/such/record.json", {std::string(OC_SOURCE_DIR)});
        Check(!none.ok && none.why == "record_missing", "(J1) a missing record reads record_missing");
    }
    {   // (J2)
        const std::string p = std::string(OC_SOURCE_DIR) + "/docs/implementation/slot16/WP-3_expected_uell.tsv";
        std::ifstream in(p);
        Check(in.good(), "(J2) " + p + " is readable");
        std::string line;
        std::vector<std::string> header;
        uint32_t rows = 0, agree = 0;
        while (std::getline(in, line)) {
            if (line.empty() || line[0] == '#') continue;
            std::vector<std::string> f;
            std::stringstream s(line);
            std::string x;
            while (std::getline(s, x, '\t')) f.push_back(x);
            if (header.empty()) { header = f; continue; }
            std::map<std::string, std::string> r;
            for (size_t i = 0; i < f.size() && i < header.size(); ++i) r[header[i]] = f[i];
            ++rows;
            bool known = false;
            const uint32_t b = census::CensusUEllBudgetMb(r["source_set"], &known);
            if (known && std::to_string(b) == r["wp3_budget_mb"]) ++agree;
        }
        Check(rows >= 230 && agree == rows, "(J2) CensusUEllBudgetMb == WP-3_expected_uell.tsv's wp3_budget_mb on every row (" +
                                                std::to_string(agree) + " / " + std::to_string(rows) + ")");
        bool k1 = false, k2 = false, k3 = true;
        Check(census::CensusUEllBudgetMb("ext_first", &k1) == 0 && k1 && census::CensusUEllBudgetMb("ext_u691", &k2) == 8192 && k2 &&
                  (census::CensusUEllBudgetMb("npx_zz9", &k3), !k3),
              "(J2) the table's fixed points: ext_first auto, ext_u691 8192, an unknown set unknown");
        uint32_t mb = 0;
        bool present = false;
        std::string err;
        Check(census::ParseUEllOverride("16384", &mb, &present, &err) && present && mb == 16384 &&
                  census::ParseUEllOverride("", &mb, &present, &err) && !present &&
                  !census::ParseUEllOverride("0", &mb, &present, &err) && !census::ParseUEllOverride("-5", &mb, &present, &err) &&
                  !census::ParseUEllOverride("1234567890", &mb, &present, &err),
              "(J2) ParseUEllOverride: 16384 read, empty = none, 0 / -5 / 10 digits refused");
        ::setenv("OC_ARM_J_CELLS_PROBE", "654:37,654-37,:7,82:7", 1);
        const census::CellList cl = census::ParseCellList("OC_ARM_J_CELLS_PROBE");
        ::unsetenv("OC_ARM_J_CELLS_PROBE");
        Check(cl.set && cl.cells.size() == 2 && cl.bad.size() == 2 && cl.bad[0].rfind("token 2 ", 0) == 0,
              "(J2) ParseCellList: '654-37' and ':7' refused by position, 654:37 and 82:7 read");
    }
    {   // (J3)
        const census::RecordSystems a = Synthetic(R"({"n_sym":10,"ell":7,"operators_computed":[2,"x"],"ideals":[]})");
        Check(!a.ok && a.why == "record_operator_not_an_integer", "(J3) a non-integer operator is refused (" + a.why + ")");
        const census::RecordSystems b = Synthetic(R"({"n_sym":10,"ell":7,"operators_computed":[2],"ideals":[{"id":1},{"id":1}]})");
        Check(!b.ok && b.why == "record_duplicate_ideal_1", "(J3) a duplicate ideal id is refused (" + b.why + ")");
        const census::RecordSystems c = Synthetic(R"({"n_sym":10,"ell":7,"ideals":[)");
        Check(!c.ok && c.why == "record_unparsable", "(J3) a truncated record is unparsable");
    }
    {   // (J4) synthetic (654, 37): the WP-12 Sec. 8.1 merged pair, ids 17 / 18, U_37 = 0 / 1
        const std::string rec =
            R"({"n_sym":654,"ell":37,"operators_computed":[2,3,5,7,37],"u_ell":{"included":true},"ideals":[)"
            R"({"id":17,"witness_poly":[9,1],"split":{"ok":true,"r":1,"primes":[2,3,5,7,37],"eigenvalues":[[18],[5],[19],[35],[0]]}},)"
            R"({"id":18,"witness_poly":[9,1],"split":{"ok":true,"r":1,"primes":[2,3,5,7,37],"eigenvalues":[[18],[5],[19],[35],[1]]}},)"
            R"({"id":38,"witness_poly":[4,0,1],"split":{"ok":true,"r":3,"primes":[2,3,5,7,37],"eigenvalues":[]}}]})";
        const census::RecordSystems rs = Synthetic(rec);
        Check(rs.ok && rs.ideals.size() == 3, "(J4) the synthetic record parses (" + rs.why + ")");
        Row t17 = {{"ideal_id", "17"}, {"witness_poly", "[9,1]"}, {"module_D", "2"}, {"residue_degree", "1"}};
        Row t18 = {{"ideal_id", "18"}, {"witness_poly", "[9,1]"}, {"module_D", "1"}, {"residue_degree", "1"}};
        Row t38 = {{"ideal_id", "38"}, {"witness_poly", "[4,0,1]"}, {"module_D", "12"}, {"residue_degree", "3"}};
        const std::vector<uint64_t> Su = {2, 3, 5, 7, 37}, Sn = {2, 3, 5, 7};
        CensusLatticeCell good;   // the re-run WITH U_37: ideals 17 / 18 apart
        good.n = 654;
        good.ell = 37;
        good.rows.push_back(RerunIdeal(17, 2, Su, {18, 5, 19, 35, 0}, {9, 1}));
        good.rows.push_back(RerunIdeal(18, 1, Su, {18, 5, 19, 35, 1}, {9, 1}));
        CensusLatticeRow f3 = RerunIdeal(38, 12, Su, {}, {4, 0, 1});
        f3.eigen_ok = false;
        good.rows.push_back(f3);
        const JoinResult j17 = JoinRow(good, t17, rs, Su, 16384), j18 = JoinRow(good, t18, rs, Su, 16384);
        Check(j17.r && j17.mode == "system" && j17.r->ideal_id == 17 && j18.r && j18.r->ideal_id == 18 && j18.r->dim_E == 1,
              "(J4) with U_37 the rows join their own ideals by system");
        const JoinResult j38 = JoinRow(good, t38, rs, Su, 16384);
        Check(j38.r && j38.mode == "witness" && j38.fallback.empty(), "(J4) an f = 3 row joins by witness, not a fallback");
        CensusLatticeCell merged;   // the re-run WITHOUT U_37 (slot 15): 17 / 18 merged into one dim-3 ideal 17
        merged.n = 654;
        merged.ell = 37;
        merged.rows.push_back(RerunIdeal(17, 3, Sn, {18, 5, 19, 35}, {9, 1}));
        const JoinResult m17 = JoinRow(merged, t17, rs, Sn, 2048);
        Check(!m17.r && m17.why.rfind("operator_set:record=[2,3,5,7,37],rerun=[2,3,5,7]", 0) == 0,
              "(J4) negative control: the merged re-run is refused operator_set (" + m17.why + ")");
        const JoinResult m38 = JoinRow(merged, t38, rs, Sn, 2048);
        Check(!m38.r && m38.why.rfind("operator_set:u_ell_record=present,rerun=absent", 0) == 0,
              "(J4) the f = 3 row's witness join checks the operator set against the record (" + m38.why + ")");
        CensusLatticeCell amb = good;   // two re-run ideals with ideal 17's system
        amb.rows.push_back(RerunIdeal(40, 2, Su, {18, 5, 19, 35, 0}, {9, 1}));
        const JoinResult a17 = JoinRow(amb, t17, rs, Su, 16384);
        Check(!a17.r && a17.why == "system_ambiguous", "(J4) an ambiguous system is refused");
        census::RecordSystems missing;
        missing.why = "record_missing";
        const JoinResult f17 = JoinRow(merged, t17, missing, Sn, 16384);
        Check(!f17.r && f17.why.rfind("operator_set:u_ell_absent_under_budget_mb=16384", 0) == 0,
              "(J4) no record, budget 16384, U_37 absent: the witness fallback is refused (" + f17.why + ")");
        Row t17d = t17;
        t17d["module_D"] = "3";
        const JoinResult g17 = JoinRow(merged, t17d, missing, Sn, 0);
        Check(g17.r && g17.mode == "witness" && g17.fallback == "record_missing",
              "(J4) no record at auto: the witness fallback joins and is reported as a fallback");
        CensusLatticeCell dup = good;   // two re-run ideals with id 38 (cannot happen in a driver run; the guard)
        dup.rows.push_back(f3);
        const JoinResult d38 = JoinRow(dup, t38, rs, Su, 16384);
        Check(!d38.r && d38.why == "witness_ambiguous", "(J4) a non-unique witness match is refused");
    }
}

void ArmAll() {
    const std::string path = CensusPath();
    const census::CellList filter_list = CellFilter();
    const std::vector<std::pair<uint32_t, uint32_t>>& filter = filter_list.cells;
    const bool join_only = EnvOn("HECKE_CENSUS_LATTICE_JOIN_ONLY");
    // (slot-14 WP-2) CELLS alone reaches here now (see the header note); say
    // which trigger actually fired so a log makes the vacuous-vs-real
    // distinction mechanically checkable, not just by counting CENSUS_LATTICE
    // rows after the fact.
    if (filter_list.set) {
        std::cout << "-- (ALL) HECKE_CENSUS_LATTICE_CELLS: " << filter.size() << " cell(s) over " << path << "\n";
    } else {
        std::cout << "-- (ALL) HECKE_CENSUS_LATTICE_ALL=1 over " << path << "\n";
    }
    for (const std::string& b : filter_list.bad)
        Check(false, "(ALL) HECKE_CENSUS_LATTICE_CELLS " + b + " is not n:ell -- refused");
    if (filter_list.set && filter.empty()) {
        Check(false, "(ALL) HECKE_CENSUS_LATTICE_CELLS names no valid cell: nothing run");
        return;
    }
    if (join_only) std::cout << "-- (ALL) HECKE_CENSUS_LATTICE_JOIN_ONLY=1: the driver and the join only (no Z/ell^2 run)\n";
    // (slot-16 WP-3) u_ell: HECKE_CENSUS_LATTICE_U_ELL=<mb> for every row, else the row's source set.
    uint32_t override_mb = 0;
    bool override_on = false;
    {
        std::string err;
        if (!census::ParseUEllOverride(std::getenv("HECKE_CENSUS_LATTICE_U_ELL"), &override_mb, &override_on, &err)) {
            Check(false, "(ALL) HECKE_CENSUS_LATTICE_U_ELL refused: " + err);
            return;
        }
    }
    // (slot-16 WP-3 audit repair, finding 2) the record roots, reported; an explicit list that names no
    // root or a missing directory is refused (the silent all-witness run of the audit).
    const census::RecordRootSet roots = census::RecordRootsFrom("HECKE_CENSUS_LATTICE_RECORDS");
    std::cout << "-- (ALL) record roots " << roots.text() << "\n";
    if (roots.explicit_empty) Check(false, "(ALL) HECKE_CENSUS_LATTICE_RECORDS is set but names no root");
    for (const std::string& m : roots.missing) {
        std::cout << "CENSUS_LATTICE_RECORD_ROOT_MISSING\t" << m << "\t" << (roots.explicit_ ? "explicit" : "default") << "\n";
        if (roots.explicit_) Check(false, "(ALL) HECKE_CENSUS_LATTICE_RECORDS names a missing directory: " + m);
    }
    const std::vector<Row> tsv = ReadTsv(path);
    // cells in census order, with their non-reduced rows
    std::vector<std::pair<uint32_t, uint32_t>> order;
    std::map<std::pair<uint32_t, uint32_t>, std::vector<const Row*>> rows;
    std::map<std::pair<uint32_t, uint32_t>, uint32_t> budget;
    std::set<std::pair<uint32_t, uint32_t>> mixed;
    uint32_t refused = 0;
    auto refuse = [&](const std::pair<uint32_t, uint32_t>& k, const Row& t, const std::string& why) {
        ++refused;
        std::cout << "CENSUS_LATTICE_REFUSED\t" << k.first << "\t" << k.second << "\t" << t.at("ideal_id")
                  << "\tline=" << t.at("#line") << "\treason=" << why << std::endl;
        Check(false, "(ALL) (" + std::to_string(k.first) + "," + std::to_string(k.second) + ") ideal "
                         + t.at("ideal_id") + " (TSV line " + t.at("#line") + ") refused: " + why);
    };
    for (const Row& t : tsv) {
        if (t.at("reduced") != "false") continue;
        const std::pair<uint32_t, uint32_t> k = {static_cast<uint32_t>(std::stoul(t.at("n"))),
                                                  static_cast<uint32_t>(std::stoul(t.at("ell")))};
        if (!filter.empty() && std::find(filter.begin(), filter.end(), k) == filter.end()) continue;
        // (slot-16 WP-3 audit repair, finding 5) the override waives the unknown-set refusal, exactly as in
        // HeckeFormsHeckeAlgebraValidation arm (T): an override names the budget, so the set's is not needed.
        bool known = true;
        const uint32_t b = override_on ? override_mb : census::CensusUEllBudgetMb(t.at("source_set"), &known);
        if (!override_on && !known) {   // never silently at the default
            refuse(k, t, "unknown_source_set:" + t.at("source_set"));
            continue;
        }
        if (!rows.count(k)) order.push_back(k);
        if (budget.count(k) && budget[k] != b) mixed.insert(k);
        budget[k] = b;
        rows[k].push_back(&t);
    }
    // (slot-16 WP-3 audit repair, finding 4) a filter cell with no non-reduced census row is reported, and a
    // filter selecting no cell at all fails (it used to print cells=0 and pass).
    for (const auto& f : filter)
        if (!rows.count(f)) std::cout << "CENSUS_LATTICE_FILTER_UNMATCHED\t" << f.first << "\t" << f.second << "\tno_non_reduced_row\n";
    if (filter_list.set && order.empty() && refused == 0)
        Check(false, "(ALL) HECKE_CENSUS_LATTICE_CELLS selects no non-reduced census row of " + path);
    std::cout << "CENSUS_LATTICE_HEADER\t" << CensusLatticeHeader() << "\n";
    CensusWeightCache weights;
    uint32_t n_rows = 0, held = 0, failed = 0, na = 0, other = 0;
    uint32_t join_system = 0, join_witness = 0, join_fallback = 0, f1_rows = 0, cells_with_uell = 0, cells_without_uell = 0;
    std::map<std::string, uint32_t> uell_cells;
    std::set<std::string> fail_rows;
    for (const auto& k : order) {
        if (mixed.count(k)) {
            for (const Row* t : rows[k]) refuse(k, *t, "mixed_u_ell_in_cell");
            continue;
        }
        bool any_h1 = false;
        for (const Row* t : rows[k]) {
            if (t->at("own_group") != "H1") {
                ++n_rows;
                ++na;
                std::cout << "CENSUS_LATTICE\t" << k.first << "\t" << k.second << "\t" << t->at("ideal_id")
                          << "\tnot_applicable:" << t->at("own_group") << "\n";
            } else {
                any_h1 = true;
            }
        }
        if (!any_h1) continue;
        CensusLatticeOptions o;
        o.u_ell_budget_mb = budget[k];
        o.join_only = join_only;
        const CensusLatticeCell c = ComputeCensusLattice(k.first, k.second, o, &weights);
        const std::vector<uint64_t> S = c.rows.empty() ? std::vector<uint64_t>() : c.rows.front().primes;
        const bool with_u = std::find(S.begin(), S.end(), static_cast<uint64_t>(k.second)) != S.end();
        (with_u ? cells_with_uell : cells_without_uell) += 1;
        ++uell_cells[census::UEllText(budget[k])];
        std::cout << "CENSUS_LATTICE_OPERATORS\t" << k.first << "\t" << k.second << "\tu_ell="
                  << census::UEllText(budget[k]) << "\tsource=" << (override_on ? "override" : "source_set")
                  << "\toperators=" << census::ListText(S) << "\tu_ell_operator=" << (with_u ? "present" : "absent") << "\n";
        if (join_only)   // the re-run's whole H^1 ideal list: the operator set's ideals, one line each
            for (const CensusLatticeRow& x : c.rows) {
                std::string a = "[";
                for (size_t i = 0; i < x.a.size(); ++i) a += (i ? "," : "") + std::to_string(x.a[i]);
                std::cout << "CENSUS_LATTICE_RERUN_IDEAL\t" << k.first << "\t" << k.second << "\t" << x.ideal_id
                          << "\tdim=" << x.dim_E << "\tsystem=" << (x.eigen_ok ? a + "]" : std::string("f>1")) << "\n";
            }
        std::map<std::string, census::RecordSystems> records;
        for (const Row* t : rows[k]) {
            if (t->at("own_group") != "H1") continue;
            ++n_rows;
            const std::string rp = t->count("record_path") ? t->at("record_path") : std::string();
            if (!records.count(rp)) {
                census::RecordSystems none;
                none.why = "no_record_path";
                records[rp] = rp.empty() ? none : census::ReadRecordSystems(rp, roots.roots);
            }
            const std::string tag = "(ALL) (" + std::to_string(k.first) + "," + std::to_string(k.second)
                                    + ") ideal " + t->at("ideal_id");
            if (!t->count("residue_degree") || t->at("residue_degree") == "1") ++f1_rows;
            const JoinResult j = JoinRow(c, *t, records[rp], S, budget[k]);
            if (!j.fallback.empty()) {
                ++join_fallback;
                std::cout << "CENSUS_LATTICE_FALLBACK\t" << k.first << "\t" << k.second << "\t" << t->at("ideal_id")
                          << "\tline=" << t->at("#line") << "\treason=" << j.fallback << "\n";
            }
            if (!j.r) {
                refuse(k, *t, "join:" + j.why);
                continue;
            }
            (j.mode == "system" ? join_system : join_witness) += 1;
            CensusLatticeRow r = *j.r;
            r.ideal_id = static_cast<uint32_t>(std::stoul(t->at("ideal_id")));   // the census's id
            std::cout << "CENSUS_LATTICE_JOIN\t" << k.first << "\t" << k.second << "\t" << r.ideal_id
                      << "\tline=" << t->at("#line") << "\tjoin=" << j.mode << "\trerun_id=" << j.r->ideal_id
                      << "\tdim=" << r.dim_E << "\tloewy_V=" << r.loewy_V << "\n";
            std::cout << CensusLatticeLine(c, r) << std::endl;
            CheckCensusAgreement(r, *t, tag);
            if (join_only || !r.eigen_ok) {   // f > 1: C-F9 not evaluated (reported, not a failure)
                ++other;
                continue;
            }
            CheckInstrument(c, r, tag);
            if (r.status == "held") ++held;
            else if (r.status == "failed") { ++failed; fail_rows.insert(tag); }
            else ++other;
        }
    }
    // (slot-16 WP-3 audit repair, finding 2) records were expected (an explicit root list) and not one
    // residue-degree-1 row found its record: the run is an all-witness run in disguise -- fail.
    if (roots.explicit_ && f1_rows > 0 && join_fallback == f1_rows)
        Check(false, "(ALL) HECKE_CENSUS_LATTICE_RECORDS was given but every one of the " + std::to_string(f1_rows) +
                         " residue-degree-1 rows fell back to the witness join (no usable record)");
    std::cout << "CENSUS_LATTICE_SUMMARY rows=" << n_rows << " held=" << held << " failed=" << failed
              << " not_applicable=" << na << " not_evaluated=" << other << " cells=" << order.size()
              << " refused=" << refused << "\n";
    for (const std::string& f : fail_rows) std::cout << "  C-F9 FAILED (a finding, not a test failure): " << f << "\n";
    // (slot-14 WP-2, plan item 5, "the vacuous-arm lesson") One EFFECTIVE line
    // per arm actually run, so a kit expect-string can be pinned to a real
    // cell count instead of to a bare rc/PASS that a vacuous arm satisfies too.
    // (slot-16 WP-3) ... and the operator set, the u_ell actually used and the join modes.
    // (slot-16 WP-3 audit repair) ... and the fallbacks and the record roots (appended: the prefix is unchanged).
    std::string u_text;
    if (override_on) {
        u_text = "override:" + census::UEllText(override_mb);
    } else {
        u_text = "source_set{";
        bool first = true;
        for (const auto& u : uell_cells) {
            u_text += (first ? "" : ",") + u.first + ":" + std::to_string(u.second);
            first = false;
        }
        u_text += "}";
    }
    std::cout << "EFFECTIVE HeckeCensusLatticeValidation arm=" << (join_only ? "JOIN" : "ALL") << " cells=" << order.size()
              << " operators=with_U_ell:" << cells_with_uell << ",without_U_ell:" << cells_without_uell
              << " u_ell=" << u_text << " join=system:" << join_system << ",witness:" << join_witness
              << ",refused:" << refused << ",fallback:" << join_fallback << " records=" << roots.text() << "\n";
}


} // namespace

uint32_t HeckeCensusLatticeValidation() {
    Check.reset();
    const auto t0 = std::chrono::steady_clock::now();
    std::cout << "HeckeCensusLatticeValidation (slot-13 WP-9: C-F9, the lattice test)\n";
    ArmP();
    ArmJ();   // (slot-16 WP-3 audit repair, finding 3) the join machinery, always on (a few ms)
    ArmS();
    // (slot-14 WP-2) CELLS alone enables the arm now, matching every other
    // _ALL/_CELLS pair in the env-hook table.  (slot-16 WP-3 audit repair) A SET CELLS variable
    // enables it even when no token parses, so a typo fails loudly instead of running nothing.
    if (EnvOn("HECKE_CENSUS_LATTICE_ALL") || CellFilter().set) ArmAll();
    const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    std::cout << "HeckeCensusLatticeValidation: " << Check.checks << " checks, " << Check.fails << " failing, " << s
              << " s\n";
    return Check.fails == 0 ? 0 : 1;
}
