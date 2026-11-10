// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

#pragma once
#ifndef HECKE_EXT_RECORD_H
#define HECKE_EXT_RECORD_H

// ===========================================================================
// hecke-extensions phases E2a / E2 cont. (slot 4 WP-1; slot 5 WP-1) -- THE
// EXTENSION RECORD'S DATA MODEL: one `HeckeExtRecord` per (n_sym, ell) cell,
// and the one `is_final`.  Schema `hecke_ext/v2`.
//
// [E] = docs/latex/hecke_extensions.tex v1.5.2 `sec:schema` (the key names
// here are ITS, verbatim, and where this file and [E] Sec. 7 differ [E] wins);
// [X] = docs/implementation/hecke_extensions_implementation_plan.md v1.8
// Sec. 1.3 (`is_final`), Sec. 1.3.1 R3.1 / R3.2, Sec. 1.4 (the writer),
// Sec. 1.5 (the settings); [C] = docs/latex/gl2_contract.tex v1.6.2
// `sec:recordschema`.
//
// THE WRITER IS A SEPARATE ONE.  `hecke_ext/v2` is emitted by
// HeckeExt/HeckeExtReport, NEVER by extending Hecke/HeckeReport's
// `hecke_report/v2`: the golden harness hashes that record's text
// (Golden/GoldenHarness.cpp's `report_canonical^q` / `report_full^q`), so a
// new key there is a golden re-baseline, and this slot's one emission is
// WP-3's ([X] Sec. 1.6; slot-5 plan Sec. 4.4).
//
// ===========================================================================
// `hecke_ext/v2` -- hecke-extensions slot 5, WP-1.  WHAT THE BUMP IS FOR.
// ===========================================================================
//
// [OD] slot-4 close-out 5 (2026-09-22) ruled the shape repair APPROVED and
// ordered it BEFORE the stage's first golden emission of an ext record, on the
// discipline `hecke_report/v1 -> v2` set in slot 2: the change alters the VALUE
// of an existing key (`resumable_items`), so it is a version bump and not an
// in-place amendment.  Five things move, and only the first is the ruling:
//
//   1  `u_ell` is a property of the CELL, not of a maximal ideal.  In v1 the
//      UEllRecord sat inside every IdealRecord, so a four-ideal cell emitted
//      one decision four times and `resumable_items()` counted one refusal
//      four times ([E] sec:schema v1.5.0's own statement of the defect).  In
//      v2 it sits at record level with ONE `resumable`, and
//      `resumable_items()` counts what it says it counts.
//   2  `groups` per ideal ([E] sec:schema v1.5.0's second booked repair):
//      which ambient groups Stage 2's cross-block evaluation test matched the
//      ideal across, with the matching witness.  `modules` keeps its "one
//      entry per ambient group" shape, so nothing else moves.
//   3  `complexes.boundary` and `complexes.pair` stop being
//      `{"status": "unavailable"}` and carry their own degree fields, the
//      cone's `les_maps` and the two RECORDED cone flags that [C]
//      `sec:recordschema` item 6(e) reserved.
//   4  The per-sign (E3) objects: `iota` per degree, the per-sign groups, and
//      an ideal's own `sign`, so that the unsplit record's ideal list is the
//      sum of the two per-sign ones.
//   5  `canonical` per ambient group, with the Eisenstein line's position.
//
// WHAT IS STILL UNAVAILABLE.  Nothing is emitted that was not computed: every
// complex, every group and every per-sign object carries the
// `{"status": "unavailable"}` placeholder of [C] `sec:recordschema` item 6(c)
// until its producer attached it.  Nothing here is a golden object: every
// collector over this record is defined and called by
// HeckeExtRecordValidation alone.
//
// [C] `sec:recordschema` item 5 -- THE DEGREE FIELDS ARE PER COMPLEX.  Every
// complex a record is about carries its own `q_max` / `truncated` /
// `q_max_source` / `degrees`, DERIVED from that complex's own degree count and
// never a maximum over the record (the `HeckeReportInput::num_degrees`
// lesson: a writer that cannot name its complex REFUSES rather than guesses).
//
// PLACEMENT.  HeckeExt/, MatrixT<uint32_t>, no hash container, no bare
// assert.  It consumes HeckeModuleStructure (hence HeckeLocalAlgebra),
// HeckeGenerationCertificates, HeckeSturmSweep, HeckeUEllBudget and
// Hecke/HeckeSettings -- all of them already HeckeExt/ or dependency-free.
// ===========================================================================

#include "HeckeExtFinal.h"
#include "HeckeGenerationCertificates.h"
#include "HeckeModuleStructure.h"
#include "HeckeSturmSweep.h"
#include "HeckeUEllBudget.h"

#include "../Hecke/HeckeSettings.h"

#include <cstdint>
#include <string>
#include <vector>

namespace hecke_ext {

// ---------------------------------------------------------------------------
// The schema strings.  ONE definition each, so that "v1 and v2 are
// distinguishable by this key alone" is a statement about a constant and not
// about a literal repeated in five places.
// ---------------------------------------------------------------------------
inline constexpr const char* kExtSchemaV1 = "hecke_ext/v1";
inline constexpr const char* kExtSchemaV2 = "hecke_ext/v2";
// (slot-7 WP-3, stage 3; [E] v1.8.0 sec:schema (h); slot-7 owner confirmation
// 10) v3 = v2 plus the stage-2 keys the default record gained in items 3-7:
// anemic_pool, s_sep, label, split, predictions_mode, predictions,
// predicted, confirmed, unpredicted (and the sturm.proof_range source list).
inline constexpr const char* kExtSchemaV3 = "hecke_ext/v3";
// (slot-12 FIN; [E] v1.13.0 sec:schema (i); owner-approved 2026-09-26) v4 =
// v3 plus the finality certificates (d0) "dickson" / (d1) "boundary":
// certificate_routes.d0 / .d1, the per-ideal boundary_reps, and the
// certificate names "dickson" / "boundary".
inline constexpr const char* kExtSchemaV4 = "hecke_ext/v4";
// (slot-13 WP-7; [E] v1.14.0-pre.1 sec:schema (j), cert:boundary; slot-12
// FIN audit G2; owner-approved 2026-09-26, slot-13 Q3) v5 = v4 plus (d1)'s
// data: `boundary_data` = {rank_r1, r1 = r^1|_V, t = {t_p|_V : p in R(ell)}}
// on every ideal with d1 = passed, so verify() (code 32) re-derives (d1)
// from the record as it re-derives (d0) (code 31).
inline constexpr const char* kExtSchemaV5 = "hecke_ext/v5";
// What this build writes.
inline constexpr const char* kExtSchema = kExtSchemaV5;

// ---------------------------------------------------------------------------
// The git commit the record claims ([E] Sec. 7 `commit`, [X] Sec. 1.4 "the git
// hash baked in by CMake, like OC_SOURCE_DIR").
// ---------------------------------------------------------------------------
//
// Returns OC_GIT_COMMIT when CMake could resolve one at configure time, else
// the literal "unknown".  It is metadata: nothing branches on it, and no
// golden or determinism digest reads it.
const char* SourceCommit();

// ---------------------------------------------------------------------------
// (slot-7 WP-3, stage 2 item 3) The anemic pool of [E] `sec:schema` (a):
// T = {primes q <= 31} \ {ell}, ascending.  ONE definition: the driver's
// cell-local `proof_range` decision computes over it, the record carries it
// as `anemic_pool`, and verify() compares the two.
// ---------------------------------------------------------------------------
std::vector<uint64_t> AnemicPool(uint64_t ell);

// (slot-7 WP-3, stage 2 item 4; owner ruling 3b (B); [E] `sec:schema` (b))
// The ells at which Sigma(ell) is built IN-PROCESS -- the range
// HeckeAtlasValidation pins -- and at no other ell: the record never reads a
// stored atlas, so golden stays self-contained.
inline bool IsInProcessAtlasEll(uint64_t ell) {
    return ell == 5 || ell == 7 || ell == 11 || ell == 13;
}

// The record-level `s_sep` object ([E] `sec:schema` (b)).  `computed` at an
// in-process atlas ell (and when the driver consulted the atlas), else
// {"status": "atlas_unavailable"} and nothing else.
struct SSepRecord {
    bool computed = false;
    std::vector<uint64_t> primes;   // S_sep(ell), ascending
    std::vector<uint64_t> added;    // the S_sep primes Stage 2(3) added to S_0, ascending
};

// (slot-7 WP-3, stage 2 item 5; [E] `sec:schema` (b)) The per-ideal atlas
// label.  Emitted as `null` on a non-local ideal (it has no character); on a
// local ideal as the computed object when `computed`, else as
// {"status": "atlas_unavailable"}.  `key` is the CanonicalLess-least
// Frobenius conjugate of the ideal's anemic system over `anemic_pool` (one
// element of F_{ell^r} per prime, each a coefficient vector of length r,
// low -> high) -- the label proper, a function of the system alone; `system`
// and `base` are ids in the in-process Sigma(ell) at the same settings,
// informative only.
struct LabelRecord {
    bool computed = false;
    std::vector<std::vector<uint32_t>> key;
    uint32_t r = 0;
    uint32_t system = 0;
    std::string type;                // "eisenstein" | "cuspidal"
    bool has_chi = false;
    uint32_t chi_a = 0, chi_b = 0;
    uint32_t base = 0;
    uint32_t theta_twist = 0;
};

// (slot-7 WP-3, stage 2 item 7; [E] `sec:schema` (d)) The cell's prediction
// INPUTS: the Kummer verdicts (ascending in k, at least k = n + 2) and the
// Glover factors of Sym^n under the standard-module convention.
struct PredictionsRecord {
    struct Kummer {
        uint32_t k = 0;
        bool applicable = false, divides = false;
        bool has_reduced = false;                 // reason direct | kummer_congruence
        uint32_t k_reduced = 0, b_over_k_mod_ell = 0;
        std::string reason;
    };
    std::vector<Kummer> kummer;
    std::string glover_json;                      // hecke_glover_factors/v1; "" -> null
};

// The closed list of prediction names ([E] `sec:schema` (d); `pr:eiscusp`).
inline constexpr const char* kPredictionEisensteinCuspidal = "eisenstein_cuspidal";

// The closed list of `sturm.proof_range_source` ([E] `sec:schema` (a)).
inline constexpr const char* kProofRangeCellLocal       = "cell_local";
inline constexpr const char* kProofRangeCallerInput     = "caller_input";
inline constexpr const char* kProofRangeMethodUndefined = "method_undefined";

// ---------------------------------------------------------------------------
// R3.1's record object, as the record layer holds it.
// ---------------------------------------------------------------------------
//
// The three shipped budgeted units (HeckeSturmBound, CommutantRoute,
// SturmSweep) already emit their own `resumable` object with the inline /
// sidecar rule; what the RECORD adds ([X] R3.1 item 6) is the STEM, which only
// it knows.  A `ResumableRecord` therefore carries the pieces and lets
// HeckeExtReport decide inline vs sidecar against the 64 KiB limit.
struct ResumableRecord {
    bool        present  = false;
    std::string kind;                        // one of HeckeExtFinal.h's five
    bool        complete = true;
    std::string budget_json         = "{}";  // the instance's own budget object
    std::string reached_json        = "{}";  // its progress counters
    std::string budget_history_json = "[]";
    std::string state_json;                  // the inline XState JSON; "" = none
};

// ---------------------------------------------------------------------------
// One complex the record is about ([C] `sec:recordschema` item 5).
// ---------------------------------------------------------------------------
// One spot of the cone's long exact sequence ([C] `eq:coneles`; [X] Sec. 6.2
// E5's `les_maps`).  It mirrors `ConeLesSpot` of EquivariantMappingCone.h
// field for field; it is re-declared here because the record layer describes
// the cone and does not build it, and HeckeExt/ owns no cone type.
struct LesMapSpot {
    std::string term;            // "Hc" | "H_bulk" | "H_bd"
    int32_t     q = 0;           // the degree of the MIDDLE term
    bool        in_built  = false;
    bool        out_built = false;
    bool        asserted  = false;   // all three terms reported
    uint32_t    dim_middle = 0, rank_in = 0, rank_out = 0;
    bool        composite_zero = false;  // meaningful iff in_built && out_built
    bool        exact          = false;  // meaningful iff asserted
    uint32_t    coker_in_dim   = 0;
};

struct ComplexRecord {
    // False => the writer emits {"status": "unavailable"} and NOTHING else
    // ([C] `sec:recordschema` item 6(c)).
    bool available = false;

    // The complex's OWN degree count.  q_max = num_degrees - 1, and the
    // writer REFUSES a record whose available complex has num_degrees == 0.
    uint32_t num_degrees = 0;
    bool     truncated   = false;
    // "set" iff a setting fixed q_max (E9's truncated model); "derived"
    // otherwise.  The writer refuses `q_max_source = "set"` with
    // `truncated = false` ([C] item 5: `set` only where truncated is true)
    // AND, since [C] v1.7.0 (slot-5 WP-11's ruling), with `derived_from`
    // non-empty -- `derived_from` non-empty IS the record's own derived
    // flag, and `"set"` is legal on a SOURCE complex only (verify() code 22).
    bool     q_max_configured = false;

    // --- v2 -------------------------------------------------------------
    // [C] `sec:recordschema` item 5, "a complex built from others": the cone
    // carries `q_max_source = "derived"` unconditionally TOGETHER WITH the
    // names of the complexes it is derived from.  Empty on a primitive one.
    // Since [C] v1.7.0 this list IS the complex's derived flag: non-empty
    // here together with `q_max_configured = true` (`q_max_source = "set"`)
    // is refused by `verify()` (code 22).
    std::vector<std::string> derived_from;

    // Labels a writer cannot derive: the boundary's model ("circle") and
    // whether it computes group cohomology (`ell != 2` on the shipped one).
    std::string model;
    bool        has_group_cohomology_flag = false;
    bool        is_group_cohomology       = false;

    // The LES spots, on the complex that IS the long exact sequence's
    // subject -- i.e. the pair.  Empty elsewhere.
    std::vector<LesMapSpot> les_maps;

    // [C] `sec:recordschema` item 6(e): the two RECORDED cone flags, reserved
    // by name in slot 4 and first emitted here.  Diagnostics under rule A6 --
    // recorded, never branched on, and a `false` is conformance.
    bool has_cone_flags     = false;
    bool phi_flat_identity  = false;
    bool residual_flat_zero = false;

    // The standalone record the cone ACTION wrote, named so that a reader can
    // find it ("hecke_cone_action/v1"); empty when no action was built.
    std::string action_record_schema;
};

// ---------------------------------------------------------------------------
// A cohomology group of [E] Sec. 7's `groups` object.
// ---------------------------------------------------------------------------
struct GroupRecord {
    std::string name;             // "H0", "H1", "H0_bd", "H1c", "H1_plus", ...
    bool        available = false;
    uint32_t    dim       = 0;
};

// ---------------------------------------------------------------------------
// [E] sec:schema's `canonical` object ([E] `alg:stage5`(4) / D6), per ambient
// group.  v1 emitted the ideal-level placeholder {"status": "unavailable"}
// because its four slots need E3 / E4 / E5 objects; v2 emits it keyed by the
// group it is about and fills the slots it has.
//
// The values are the short prose tokens of [E] Sec. 7's example, which is what
// makes them comparable across records; `"n/a"` means the slot does not apply
// to this cell and `"unavailable"` that nothing computed it.
// ---------------------------------------------------------------------------
struct CanonicalPosition {
    std::string group;                          // the ambient group's name
    std::string eisenstein = "unavailable";
    std::string H1_shriek  = "unavailable";
    std::string dickson    = "n/a";
    std::string L_image    = "n/a";
};

// ---------------------------------------------------------------------------
// [E] sec:schema v1.5.0's SECOND booked repair (slot-5 plan Sec. 2.3 item 3):
// an ideals[] entry is per (group, block), and v1 gave no key that REPORTED
// the cross-block matching.  One entry per ambient group the ideal was tested
// against, with the witness the test used ([XA M5], `SameMaximalIdeal`).
// ---------------------------------------------------------------------------
struct IdealGroupMatch {
    std::string group;                 // the ambient group's name
    bool        own     = false;       // the group this entry's block lives in
    bool        matched = false;       // the evaluation test said yes
    uint32_t    ideal_id = 0;          // the entry the test was run FROM
    // The witness of the ideal the test was run FROM: pi_w low -> high, its
    // exponent N, and the N' the test actually used on THIS block.
    std::vector<uint32_t> witness_poly;
    uint32_t    w_hat_exponent       = 0;
    uint32_t    w_hat_exponent_other = 0;
    uint32_t    failure              = 0;   // EvaluationFailure; 0 iff matched
};

// ---------------------------------------------------------------------------
// The per-ideal sub-objects.
// ---------------------------------------------------------------------------

struct CoefficientFieldRecord {
    bool     present         = false;   // emitted only when `local` is true
    uint32_t w_hat_exponent  = 0;
    std::vector<uint64_t>              primes;   // the keys of `P`
    std::vector<std::vector<uint32_t>> P;        // index-aligned, low -> high
};

// [E] Sec. 7's `sturm` object is EXACTLY {B_prime, cap, closure_dim, dim_M_k};
// `mode`, `outcome`, `sturm_primes` and `resumable` are its siblings.
//
// `proof_range` / `proof_range_source` ([E] `sec:schema` (a), v1.8.0-pre.1;
// slot-7 WP-3, stage 2 item 3; owner ruling 3a).  `proof_range` is the
// Dickson-type test of [E] `cl:sturmmodl` alone; the source is one of the
// closed list
//   "cell_local"       -- DECIDED at the cell (HeckeExtDriver, the default):
//                         the ideal's anemic system over `anemic_pool` equals
//                         none of the cell's H^0 systems over the same pool;
//   "caller_input"     -- supplied by the caller (ExtDriverOptions::
//                         proof_range) and not verified;
//   "method_undefined" -- no recorded B'(k, ell): `proof_range` is null
//                         (`has_proof_range = false`), and `final` is
//                         governed by `sturm_unavailable` as before.
// The retired "dickson_count" (every v2 record through slot 6 -- a caller
// input under another name) is refused by verify() code 25.  The decision
// reaches `final` only through the Sturm route's outcome, never through a
// flag flipped after the fact ([C] rule A6; slot-6 WP-9 finding 4).
struct SturmRecord {
    bool     present     = false;
    bool     has_B_prime = false;
    uint32_t B_prime     = 0;
    uint32_t cap         = 0;
    uint32_t closure_dim = 0;
    uint32_t dim_M_k     = 0;

    std::string mode    = "nonfinal";     // ToString(SturmUnavailableMode)
    std::string outcome = "in_progress";  // ToString(SturmSweepOutcome)
    std::vector<uint64_t> sturm_primes;

    bool        has_proof_range    = true;    // false iff emitted as `null`
    bool        proof_range        = true;
    std::string proof_range_source = "caller_input";

    ResumableRecord resumable;
};

// [X] Sec. 1.5 / Sec. 2.2 Stage 1: the `u_ell` verdict as recorded.
struct UEllRecord {
    bool        present       = false;
    bool        included      = false;
    std::string reason;                    // within_budget|u_ell_over_budget|off
    double      predicted_mb  = 0.0;
    uint32_t    budget_mb     = 0;
    std::string model_version;
    ResumableRecord resumable;             // R3.3 item 4 (filled by WP-2)
};

struct CertificateRoutes {
    std::string a = "not_run";   // cyclic | not_cyclic | not_run
    std::string b = "not_run";   // ToString(CommutantOutcome) | not_run
    std::string c = "not_run";   // found | none_at_cap | not_applicable | off | not_run
    // (slot-12 FIN, hecke_ext/v4) The finality certificates of [E] sec:certgen,
    // run only when (a)-(c) did not pass (HeckeFinalityRoutes.h):
    //   d0: passed | failed:<class> | not_applicable:u_ell | not_run   (X_1 blocks)
    //   d1: passed | not_injective | failed:<p> | not_applicable:u_ell | not_run (X_4)
    std::string d0 = "not_run";
    std::string d1 = "not_run";
};

struct ModuleRecordEntry {
    std::string     group;       // the ambient group's name, e.g. "H1"
    ModuleStructure structure;
};

// ---------------------------------------------------------------------------
// The inputs `is_final` reads -- and nothing else ([C] rule A6: a record field
// is never branched on by the pipeline that wrote it, so `is_final` reads
// these, not the emitted strings).
// ---------------------------------------------------------------------------
struct FinalInputs {
    // [E] Def. 2.8: dim M_k == 0 (n = 0, odd n, k < 4).  There is no claim to
    // be final about, so the record says so under its own reason.
    bool not_applicable = false;

    // Locality ([X] R3.3 item 2).  `locality_complete == false` is the budget
    // survivor, which budget mode never PROVES non-local; `not_local` is
    // reachable only through strict mode's interruption path.
    bool local             = false;
    bool locality_complete = true;

    // A cooperative interruption at a step boundary (R3.3 item 5).
    bool interrupted = false;

    // Every applicable certificate's verify() returned 0.
    bool     certificates_ok    = true;
    uint32_t first_failing_code = 0;

    // The three generation routes.
    bool             cyclic_passed = false;
    CommutantOutcome commutant     = CommutantOutcome::InProgress;

    SturmSweepOutcome sturm             = SturmSweepOutcome::InProgress;
    bool              sturm_proof_range = true;
    bool              sturm_has_B_prime = false;

    // (slot-12 FIN) The finality certificates (d0) / (d1) of [E] sec:certgen:
    // a passed route makes the ideal final exactly as (a)-(c) do.
    bool dickson_passed  = false;
    bool boundary_passed = false;
};

// The tri-state of [X] Sec. 1.3.1 R3.2 over the mathematical settings in
// force.  The GENERATION half delegates to the already-validated
// `generation_verdict` of HeckeGenerationCertificates, so E2's verdict and
// E1a/E1b's cannot drift apart; what this adds is `not_applicable`,
// interruption, locality, a failed certificate, and the two heuristic modes:
//
//   * `sturm_unavailable = heuristic` turns a complete sweep with NO B'
//     (cap hit or none) into `"heuristic"`;
//   * `final_mode = sturm_heuristic` turns a complete sweep OFF THE PROOF
//     RANGE ([E] `cert:sturm`'s heuristic range) into `"heuristic"`.
//
// In the DEFAULT configuration (`closure` + `nonfinal`) neither fires and
// `"heuristic"` is unreachable ([X] R3.2, slot-1 confirmation 8).
Finality is_final(const FinalInputs& in, const hecke::MathSettings& m);

// ---------------------------------------------------------------------------
// One maximal ideal.
// ---------------------------------------------------------------------------
struct IdealRecord {
    uint32_t id  = 0;
    uint32_t dim = 0;                    // dim_{F_ell} V

    // v2 (E3): which side of the involution's split this entry belongs to.
    // "all" on the unsplit record; "plus" / "minus" on a per-sign one.  The
    // unsplit record's ideal list is the SUM of the two per-sign lists
    // ([X] Sec. 6.2 E3 acceptance (c)).
    std::string sign = "all";

    uint32_t residue_degree           = 0;
    bool     residue_degree_certified = false;
    bool     local                    = false;

    // (p, c_p) of the certifying linear combination w = sum c_p t_p.
    std::vector<std::pair<uint64_t, uint32_t>> residue_witness;
    std::vector<uint32_t>                      witness_poly;   // pi_w, low -> high
    CoefficientFieldRecord                     coefficient_field;

    // (slot-7 WP-3, stage 2 item 5) The atlas label; always emitted
    // (null / atlas_unavailable / computed, see LabelRecord).  verify() code 27.
    LabelRecord                                label;

    // (slot-7 WP-3, stage 2 item 6; [E] `sec:schema` (c), `sec:field` tier
    // (ii); [OD] OQ10, E6-3) The canonical split: the
    // `hecke_canonical_split/v1` fragment of `alg:split` VERBATIM, as
    // CanonicalSplit::to_json() writes it.  Present iff the ideal is local
    // (carries `coefficient_field`) AND split_field_report = canonical;
    // `split_ok = false` is a certificate failure, refused by verify() code 28.
    bool        split_present = false;
    bool        split_ok      = false;
    uint32_t    split_r       = 0;
    std::string split_fragment;

    // (slot-7 WP-3, stage 2 item 7; [E] `sec:schema` (d)) Always emitted.
    // `predicted`: prediction ids "<name>:<argument>" from the closed list;
    // `confirmed`: index-aligned, 1 true / 0 false (refuted) / -1 null (not
    // evaluated); `unpredicted`: the entry's ambient groups (by their record
    // names) with m X_{i,m} != 0 that no prediction covers.
    std::vector<std::string> predicted;
    std::vector<int8_t>      confirmed;
    std::vector<std::string> unpredicted;

    // Strict locality mode only ([WP3-3] finding 6, resolved by NAMING BOTH):
    // the per-block phase; the run's high-water mark is a record-level field.
    bool        has_phase_reached = false;
    std::string phase_reached;

    // The local algebra's invariants ([E] Sec. 7's `algebra` object).
    bool                  algebra_present = false;
    uint32_t              algebra_dim     = 0;
    uint32_t              embdim          = 0;
    uint32_t              nilpotency      = 0;
    bool                  gorenstein      = false;
    std::vector<uint32_t> hilbert;

    // Generation.
    std::string           certificate = "unavailable";  // cyclic|commutant|sturm(B')|unavailable
    CertificateRoutes     certificate_routes;
    std::vector<uint64_t> generators_certified;
    // (slot-12 FIN, hecke_ext/v4) R(ell), the class representatives (d1)
    // TESTED -- never generators (proposal v2 Sec. 3.4 item 2): non-empty iff
    // d1 is passed or failed:<p>, then the least prime of every class of
    // (Z/ell)^x in class order.
    std::vector<uint64_t> boundary_reps;
    // (slot-13 WP-7, hecke_ext/v5; [E] v1.14.0-pre.1 cert:boundary) the data
    // (d1) decided on, present IFF certificate_routes.d1 == "passed": r1 =
    // r^1|_V = r^1 V (h^1(bd) x D; rows in the stored basis of H^1(bd),
    // columns in the block basis of the stored t_p), rank_r1 its rank, and t
    // the restrictions t_p|_V (D x D, same block basis), index-aligned with
    // boundary_reps.  Tested, never generators; never an input to `final`.
    struct BoundaryData {
        bool                   present = false;
        uint32_t               rank_r1 = 0;
        ExtMatrix              r1;
        std::vector<ExtMatrix> t;
    };
    BoundaryData          boundary_data;
    SturmRecord           sturm;
    // v2: `u_ell` is NOT here.  It is a property of the cell and lives at
    // record level, with ONE `resumable` ([OD] slot-4 close-out 5).

    // The tangent table over K: index-aligned prime -> "active"/"inactive".
    std::vector<uint64_t>    tangent_primes;
    std::vector<uint8_t>     tangent_active;

    // The stored t_p|_V, index-aligned.
    std::vector<uint64_t>  operator_primes;
    std::vector<ExtMatrix> t_p;

    // Stage 5, one entry per ambient group.
    std::vector<ModuleRecordEntry> modules;

    // v2: Stage 2's cross-block matching, and the canonical submodules of D6,
    // both keyed by the ambient group.
    std::vector<IdealGroupMatch>   groups;
    std::vector<CanonicalPosition> canonical;

    // Any further resumable items the ideal carries (locality, commutant, ...);
    // the Sturm and u_ell ones live in their own objects above.
    std::vector<ResumableRecord> resumables;

    // The verdict and its inputs.  `final` is what `is_final(final_inputs, m)`
    // returned; the writer emits it and never recomputes it, and nothing in
    // the pipeline branches on it ([C] rule A6).
    FinalInputs final_inputs;
    Finality    final;

    // (slot-7 WP-2; [X] Sec. 2.2 Stage 5, OQ11; [E] `sec:schema`, slot-7
    // owner confirmation 15) The on-request presentation, present ONLY under
    // `presentation = on`.  `presentation_fragment` is HeckePresentation's
    // `hecke_presentation/v1` text verbatim; the writer wraps it as
    // {"generator_basis":"block","relation_basis":"algebra_rref",
    //  "fragment":<it>} -- generator indices are BLOCK-basis indices (the
    // stored basis of V = X_{i,m} the Nakayama lift is taken in), relations
    // are t-tuples over the RREF basis of A_S.  Not a hash input.
    bool        presentation_present = false;
    std::string presentation_fragment;

    // Every `resumable` the ideal carries, complete or not.
    std::vector<const ResumableRecord*> all_resumables() const;
};

// ---------------------------------------------------------------------------
// The record.
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// The iota^* datum of ONE degree ([C] `sec:involution`, `prop:iotasplit`(1);
// phase E3).  v1 carried only the record-level constant `iota_sign`, which
// names the NORMALISATION; these are the computed objects beside it.
// ---------------------------------------------------------------------------
struct IotaDegreeRecord {
    uint32_t degree = 0;
    // "split" (2 in R^x) | "not_applicable" (ell = 2, [C] `prop:iotasplit`(2)).
    std::string status = "split";

    uint32_t dim_H = 0, dim_plus = 0, dim_minus = 0;
    uint32_t rank_iota = 0;
    std::vector<uint32_t> charpoly_iota;    // low -> high (A-N4)

    // [C] `obl:iota`(v): asserted at ell >= 5, recorded at ell = 3.
    bool     commutation_asserted = false;
    uint32_t operators            = 0;
    uint32_t operators_commuting  = 0;

    // [E] D6 -- the Eisenstein line soc(X^+_{4,m}) on the PLUS side.
    bool        eisenstein_applicable      = false;
    // "defined" | "undefined" | "not_applicable"
    std::string eisenstein_status          = "not_applicable";
    uint32_t    eisenstein_block           = 0;
    uint32_t    eisenstein_f               = 0;
    uint32_t    eisenstein_dim_socle       = 0;
    uint32_t    eisenstein_dim_socle_over_F = 0;
    bool        eisenstein_socle_equals_mV = false;
};

// ---------------------------------------------------------------------------
// [C] `obl:iota`(ii)'s RIDER ([C] v1.7.0; [OD] 2026-09-22 Q3).
// ---------------------------------------------------------------------------
// The clause's EXACT Mat22 half runs only where exact relation data was
// supplied (`HeckeInvolution::set_exact_relation_data`), and the rider
// requires that an implementation "records which rows it ran it on, so that a
// record says whether the exact comparison happened rather than leaving a
// reader to assume it".  `InvolutionCertificate` has carried it per row since
// slot 4 -- `exact_relation_data` plus `RelationRow::has_exact` -- but NO
// EMITTED RECORD did, which is precisely what left the rider half discharged
// (slot-5 WP-12 Q3: assertion half yes, recording half no).  This object is
// the record's half, and `hecke_ext/v2` is where it goes, `ι` appearing in no
// other emitted record.
//
// PLACEMENT ([C] `sec:recordschema` item 5's rule -- a field belongs to the
// object it is a statement about).  This is a property of the CELL, not of a
// degree: the relation rows are the local system's, and ONE HeckeInvolution
// discharges obl:iota(ii) once for the whole complex.  So it sits beside
// `iota` at record level and not inside an `IotaDegreeRecord`, exactly as
// `u_ell` sits at record level and not inside an `IdealRecord`.
//
// Recorded, never branched on ([C] rule A6).
// ---------------------------------------------------------------------------
struct IotaExactComparisonRecord {
    // The involution was computed, so the rider applies and the writer emits
    // this object.  `verify()` REFUSES a record that carries `iota` degree
    // objects with this absent (code 23) -- the rider, enforced.
    bool present = false;

    // `InvolutionCertificate::exact_relation_data`: the exact Mat22 half ran
    // at all.  `false` is conformance, not a failure -- it means the fibre
    // half alone was available -- and the whole point of the rider is that a
    // reader can tell the two apart instead of assuming.
    bool ran = false;

    // Every relation row obl:iota(ii) walked, and the subset that carried an
    // exact Mat22 (`RelationRow::has_exact`).
    uint32_t relation_rows   = 0;
    uint32_t rows_with_exact = 0;

    // WHICH rows the exact comparison ran on: ascending, strictly increasing
    // indices into the certificate's relation list.  A row is compared iff
    // the exact half ran AND it and its matched image row both carry an exact
    // Mat22 -- the predicate `InvolutionCertificate::verify()` clause 6 uses,
    // and nothing else.
    std::vector<uint32_t> rows_compared;
};

struct HeckeExtRecord {
    std::string schema     = kExtSchema;
    std::string commit;                               // SourceCommit() if empty
    std::string model      = "sl2z_7vertex";
    // The constant of [C] `rmk:iotasign` / [E] Sec. 7: it NAMES the
    // normalisation and is true by definition; never a free parameter, never
    // branched on.  The COMPUTED per-degree objects are `iota` below.
    std::string iota_sign  = "eisenstein_in_H1_plus";

    uint32_t    n_sym  = 0;
    uint64_t    ell    = 0;
    uint32_t    weight = 0;                           // k = n_sym + 2
    // "all" on the unsplit record, "plus" / "minus" on a per-sign one.
    std::string sign   = "all";

    hecke::MathSettings settings;
    std::string         coefficient_ring = "F_ell";

    // Recorded, never gated ([E] Sec. 7 `convention_check`).
    bool convention_present      = false;
    bool tau_mod_ell_ok          = false;
    bool eis_eigenvalue_ok       = false;

    // One object per complex; only `bulk` is available before slot 5.
    ComplexRecord bulk;
    ComplexRecord boundary;
    ComplexRecord pair;

    std::vector<GroupRecord> groups;
    uint32_t dim_M_k      = 0;
    uint32_t dim_S_k      = 0;
    uint32_t dim_dickson  = 0;
    // [E] sec:schema v1.5.0: `groups` holds groups AND a CLOSED LIST of four
    // integer scalars, the fourth of which is dim H^1_! = dim ker r^1.  It is
    // emitted as `null` where the boundary complex is unavailable, so that the
    // closed list is a constant of the schema and not a function of the cell.
    bool     has_dim_H1_shriek = false;
    uint32_t dim_H1_shriek     = 0;

    // v2 (E3): one entry per degree the involution was computed in; empty
    // where it was not computed.
    std::vector<IotaDegreeRecord> iota;

    // v2 ([OD] 2026-09-22 Q3): [C] `obl:iota`(ii)'s rider -- whether the exact
    // Mat22 comparison happened and on which rows.  REQUIRED whenever `iota`
    // is non-empty; `verify()` returns 23 otherwise.
    IotaExactComparisonRecord iota_exact;

    // v2: the cell-level U_ell verdict, with ONE `resumable`
    // ([OD] slot-4 close-out 5; [E] sec:schema v1.5.0's booked repair).
    UEllRecord u_ell;

    // The long exact sequence's summary ([E] Sec. 7 `les`), unavailable here.
    bool les_available      = false;
    bool les_exact          = false;
    bool les_hecke_equivariant = false;
    std::vector<std::pair<std::string, uint32_t>> les_ranks;

    std::vector<uint64_t>    operators_computed;

    // (slot-7 WP-3, stage 2 item 3; [E] `sec:schema` (a)) The anemic pool
    // T = {q <= 31} \ {ell}, ascending, over which the cell-local
    // `proof_range` decision compares systems -- AnemicPool(ell).  Emitted
    // always (as `[]` on a hand-built record that never set it); verify()
    // code 25 requires it to equal AnemicPool(ell) wherever an ideal's source
    // is "cell_local".  `operators_computed` keeps its meaning.
    std::vector<uint64_t>    anemic_pool;

    // (slot-7 WP-3, stage 2 item 4) Stage 2(3)'s separating set, always
    // emitted; verify() code 26.
    SSepRecord               s_sep;

    // (slot-7 WP-3, stage 2 item 7) The prediction inputs; `predictions_mode`
    // is written from `settings.predictions`.  verify() code 29.
    PredictionsRecord        predictions;

    std::vector<IdealRecord> ideals;

    uint32_t identities_passed = 0;
    uint32_t identities_failed = 0;
    std::vector<std::string> certificates_ran;
    std::vector<std::pair<std::string, uint64_t>> hashes;

    // Strict locality mode only: the run's high-water phase ([WP3-3] finding 6).
    bool        has_phase_reached_high_water = false;
    std::string phase_reached_high_water;

    // Every `resumable` the RECORD itself carries (today: `u_ell`).
    std::vector<const ResumableRecord*> all_resumables() const;

    // The count of INCOMPLETE resumable items ([X] R3.1's `resumable_items`).
    //
    // v2: the record's own items PLUS each ideal's.  In v1 a cell-level
    // refusal of U_ell was attached to every ideal and counted once per
    // ideal, so a four-ideal cell reported 4 for one decision; the count is
    // now the number of incomplete items the record actually carries
    // ([C] `sec:recordschema` item 6(b) is what made that legal but
    // unreadable, and [E] sec:schema v1.5.0 booked the repair).
    uint32_t resumable_items() const;

    // 0 on success, else the first failing clause.  Structural only: the
    // per-complex rule, the reason vocabulary, index alignment, the settings
    // fingerprint's agreement with `settings`, and every embedded
    // ModuleStructure::verify().  It does NOT re-run the mathematics -- that
    // is each embedded certificate's own verify().
    uint32_t verify() const;
    static const char* verify_message(uint32_t code);
};

} // namespace hecke_ext

#endif // HECKE_EXT_RECORD_H
