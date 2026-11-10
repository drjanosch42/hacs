// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

#pragma once
#ifndef HECKE_EXT_DRIVER_H
#define HECKE_EXT_DRIVER_H

// ===========================================================================
// hecke-extensions phase E2b (slot 4, WP-2) -- THE STAGE 1-5 ORCHESTRATION.
//
// [X] = docs/implementation/hecke_extensions_implementation_plan.md v1.7
// Sec. 2.2 (Stage -> code unit), Sec. 1.3.1 R3.3 item 4 (the `u_ell` resume),
// Sec. 1.4 / Sec. 1.5; [E] = docs/latex/hecke_extensions.tex v1.5.2 (`sec:alg`
// Stages 1-5, `sec:schema`); [C] = docs/latex/gl2_contract.tex v1.5.2, cited
// by label.
//
// WHAT THIS IS.  Slot-4 WP-1 shipped the extension RECORD and every piece that
// fills it; what it did not ship is the thing that RUNS a cell.
// `HeckeExtDriver` is that: given `(n_sym, ell)` it owns one triangulation,
// one local system, ONE computed cohomology and ONE `HeckeOperatorFamily`, and
// walks
//
//   Stage 1  S_0 = {2, 3, 5, 7} u {ell iff UEllDecision admits U_ell}
//            (HeckeExt/HeckeSturmSweep's ComputeStageOneOperators) and the
//            family's A^{(q)}(T_p) for every p in S_0;
//   Stage 2  hecke::HeckeModuleDecomposition on that family, per group
//            (the bulk degrees with dim H^q > 0), with the E0b locality
//            refinement inside compute() and certify() re-derived;
//   Stage 3  HeckeExt/HeckeLocalAlgebra per block -- CONSUMING the
//            decomposition's own certified PrimaryBlock locality data
//            (`witness_poly`, `w_hat_exponent`, `character`, the witness),
//            never re-deriving it differently ([X] Sec. 2.2 Stage 3; the
//            slot-4 plan Sec. 2.3 item 2 makes this the package's performance
//            requirement, see `ExtDriverOptions::consume_block_locality`);
//   Stage 4  certificates (a) cyclicity, (b) commutant, (c) the Sturm sweep,
//            combined by `generation_verdict_with_sweep`;
//   Stage 5  HeckeExt/HeckeModuleStructure;
//
// and then fills WP-1's `HeckeExtRecord` and its `is_final` verdict.
//
// ONE PIPELINE ALIVE AT A TIME.  The family's `pipeline_cache_` never evicts,
// so only the primes of S_0 ever go through it; a Sturm prime q outside S_0 is
// STREAMED through HeckeSturmSweep's `StreamedOperatorSupplier`, which builds
// its coset / tau / Theta pipeline in a local scope, keeps the D x D
// restriction and destroys everything else.  The driver decides reuse vs
// stream (`ReuseBeforeStreamingSupplier`), and the streamed cost is recorded
// (`streamed()`), never inferred.
//
// THE `u_ell` RESUME ([X] Sec. 1.3.1 R3.3 item 4).  A refused U_ell leaves the
// record's CELL-LEVEL `u_ell` object with ONE `resumable {kind: "u_ell"}`
// (slot-5 WP-1, `hecke_ext/v2`; in v1 it sat on every ideal).
// `resume_u_ell(budget_mb)` re-runs Stage 1 with U_ell attached to the SAME
// family -- whose cache is keyed by (p, r), so the other operators and the
// cohomology are reused -- and then Stages 2-5.  A budget at or below the one
// already reached is refused with a named error; a resume on an INCLUDED
// verdict is a strict no-op (neither the budget nor the history moves).  The
// continuation is a prefix cut because the decision is a pure function of
// (n, ell, budget_mb) and Stage 2 orders operators by ascending prime, so the
// enlarged run is the run `S_0 u {ell}` would have produced from the start --
// which `HeckeExtRamanujanValidation`'s resume arm checks entrywise.
//
// PLACEMENT.  HeckeExt/, `MatrixT<uint32_t>` throughout.  Like
// HeckeSturmSweep.cpp this TU includes Hecke/ pipeline headers; it includes no
// Engine/, Cuda/ or Console/ header and no BlockFlatMatrix.h directly (guard
// rule 2), and no `unordered_` container (rule 4).  It is single-threaded: the
// concurrency in a campaign is one driver per job.
// ===========================================================================

#include "HeckeExtComplexRecords.h"
#include "HeckeExtRecord.h"
#include "HeckeFinalityRoutes.h"
#include "HeckePresentation.h"
#include "HeckeSturmSweep.h"

#include "../Hecke/HeckeInvolution.h"
#include "../Hecke/SL2ZBoundaryPair.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace hecke_ext {

// ---------------------------------------------------------------------------
// Options -- everything that is NOT a mathematical setting.
// ---------------------------------------------------------------------------
//
// The mathematical settings (`u_ell`, `sturm`, `sturm_unavailable`,
// `locality`, `commutant_budget`, `final_mode`) are read from
// `hecke::math_settings()` in force and are RECORDED, never taken from here
// ([OD] R1: the engine owns them).  These are engineering knobs only, and none
// of them can change a computed value -- `run_commutant` / `run_sturm` change
// which ROUTES are attempted, which the record states.
struct ExtDriverOptions {
    // Empty = every bulk degree q with dim H^q > 0, ascending.  The group name
    // in the record is "H<q>".
    std::vector<uint32_t> degrees;

    // Empty = DefaultGoodPrimes() = {2, 3, 5, 7}.  `ell` is never named here:
    // it enters S_0 only through the `u_ell` verdict.
    std::vector<uint64_t> good_primes;

    // Stage 4 routes.  (a) always runs (it is part of the LocalAlgebra).
    bool run_commutant = true;
    bool run_sturm     = true;
    // (slot-12 FIN) the finality certificates (d0) / (d1) of [E] sec:certgen
    // (HeckeFinalityRoutes.h).  Off: both record `not_run` -- the switch that
    // lets a validation study route (c)'s own verdict, like run_sturm for (c).
    bool run_finality  = true;

    // Stage 4(c) "reuse before streaming": a Sturm prime outside S_0 is built
    // by the streamed route.  With streaming off such a prime is simply
    // unavailable and the sweep records it as such.
    bool allow_streaming = true;

    // [E] `cl:sturmmodl`'s range.
    //
    // (slot-7 WP-3, stage 2 item 3; owner ruling 3a; [E] `sec:schema` (a))
    // UNSET, the default: the range is DECIDED AT THE CELL, at every ell --
    // the ideal's anemic system over AnemicPool(ell) is compared with the
    // cell's own H^0 systems over the same pool (`tors H^1(Z) (x) F_ell ~=
    // H^0`), and `proof_range = true` iff it equals none of them.  The
    // decision is fed to the Sturm SWEEP (it reaches `final` only through the
    // sweep's outcome, slot-6 WP-9 finding 4) and recorded with source
    // "cell_local"; without a recorded B'(k, ell) the method is undefined and
    // the record says "method_undefined" with `proof_range = null`.
    //
    // SET: the caller's value is used unverified and recorded as
    // "caller_input" -- through slot 6 this was the only mode (default true,
    // recorded as "dickson_count").  Validation arms that force a branch use it.
    std::optional<bool> proof_range;

    // Stage 3 fast path -- the slot-4 plan Sec. 2.3 item 2 / [X] Sec. 2.2
    // Stage 3.  True (the default): the local algebra CONSUMES the certified
    // `PrimaryBlock` locality data the decomposition loop already produced.
    // False: `ComputeLocalAlgebra` re-runs `hecke::CertifyLocality` from
    // scratch, which is the pre-E2b behaviour and the validation's oracle --
    // the two must agree entrywise, which is what makes the fast path a
    // performance change and not a mathematical one.
    bool consume_block_locality = true;

    // --- the `hecke_ext/v2` attachments (slot-5 WP-1) ---------------------
    //
    // BOTH OFF BY DEFAULT.  A default cell run therefore costs exactly what it
    // cost in slot 4, and neither flag can change a value the record already
    // carried: they only ADD objects, each under the placeholder rule of
    // [C] `sec:recordschema` item 6(c).
    //
    // `with_cone` builds Hecke/SL2ZBoundaryPair and its EquivariantMappingCone
    // on the SAME cohomology the rest of the driver uses, and fills
    // `complexes.boundary`, `complexes.pair` (with its `les_maps`), the
    // boundary and cone groups, `les` and `dim_H1_shriek`.
    bool with_cone = false;

    // `with_sign_split` builds the PRODUCTION Hecke/HeckeInvolution and, per
    // degree, HeckeExt/HeckeSignSplit, and fills the `iota` objects, the
    // per-sign groups, the Eisenstein line's CanonicalPosition and the two
    // per-sign RECORDS of [X] Sec. 6.2 E3 acceptance (c).
    bool with_sign_split = false;

    // (slot-24 G6) Memory only: once build_cone_() has attached the cone's
    // record objects, release the cone's cochain-level objects
    // (EquivariantMappingCone::release_cochain_objects), so they are not alive
    // during the iota phase.  cone() then serves only dimensions, maps, ranks
    // and spots (attach_cone_action() and certificate() need the objects).
    // Default off; the census campaign (CampaignExt) sets it under the
    // HECKE_FIELD_DIET lever (Util/FieldDiet.h).
    bool release_cone_cochains = false;

    // (slot-6 WP-2, OQ11) The ON-REQUEST PRESENTATION of [X] Sec. 2.2
    // Stage 5: generators and relations of every certified-local ideal over
    // T_m (HeckeExt/HeckePresentation), and -- when `with_sign_split` also
    // ran -- of every per-sign ideal.  OFF BY DEFAULT, and it ONLY computes:
    // the `hecke_ext/v2` record carries no `presentation` object in slot 6
    // (its settings key, its record key and the hash-invariance arm are slot
    // 7's re-baseline, slot-6 plan Sec. 2.3 item 2), so the record is
    // byte-identical with the flag on or off.  Read the result through
    // `presentation()` / `sign_presentation()`.
    //
    // (slot-7 WP-2) The SETTINGS KEY `presentation = on` (Hecke/HeckeSettings)
    // is what puts the object INTO the record: it computes as this flag does
    // AND attaches a `presentation` object to every ideal of the record and of
    // the per-sign records ([E] `sec:schema`).  This flag alone still only
    // computes, so a record is byte-identical with it on or off.
    bool with_presentation = false;

    // (slot-6 WP-9) [E] `alg:decomp` Stage 2(3): the atlas's separating set
    // S_sep(ell) ([E] `cl:sep`, HeckeExt/HeckeAtlas's greedy `s_sep`).  (Slot
    // 6: empty by default; see the slot-7 paragraph below.)  Non-empty: Stage 1
    // runs on S_0 u S_sep (ell removed exactly as for `good_primes`), which is
    // the re-decomposition "add S_sep(ell) to S_0 and repeat" -- a run from
    // the start on the union is that repeat, because Stage 2 orders operators
    // by ascending prime and a finer operator set only refines (the same
    // prefix argument as `resume_u_ell`).  NOT a settings key (slot-6 plan
    // Sec. 2.3 item 2) and not reachable from the campaign or the console in
    // slot 6: the record states what it ran on (`operators_computed`), and
    // HeckeExtEndToEndValidation measures which cells it would split.
    //
    // (slot-7 WP-3, stage 2 item 4; owner ruling 3b (B); [E] `sec:schema`
    // (b)) UNSET, the default: at an in-process atlas ell (5, 7, 11, 13) the
    // driver reads S_sep(ell) from the in-process Sigma(ell)
    // (HeckeAtlas::InProcessAtlas, built once per (ell, settings)) and runs
    // Stage 1 on S_0 u S_sep; elsewhere nothing is added.  SET (even empty):
    // exactly the given primes are added -- the validation arms' override.
    std::optional<std::vector<uint64_t>> s_sep;

    // (slot-7 WP-3) Whether the driver may consult the in-process atlas at
    // all (S_sep, and the item-5 labels).  BuildAtlas's own cell runs set it
    // false -- the atlas is built FROM driver runs and must not recurse.
    // With it false every atlas-derived record object is `atlas_unavailable`.
    bool consult_atlas = true;

    // Prints one `EXT_DRIVER ...` line per stage.  Diagnostic only.
    bool trace = false;
};

// ---------------------------------------------------------------------------
// The phase split -- exactly the three quantities [X] Sec. 6.2 E1 (d), as the
// slot-4 plan Sec. 2.3 item 2 restates it, is a statement about.
// ---------------------------------------------------------------------------
struct ExtDriverTimings {
    double setup_s = 0.0;   // K + model + rho_n relations + compute_cohomology()
    double stage1_s = 0.0;  // S_0 and the family's A^{(q)}(T_p) for p in S_0
    double stage2_s = 0.0;  // HeckeModuleDecomposition::compute() + certify()
    double stage3_s = 0.0;  // ComputeLocalAlgebra per block
    double stage4_s = 0.0;  // certificates (a), (b), (c)
    double stage5_s = 0.0;  // ComputeModuleStructure per block
    double record_s = 0.0;  // filling the record and is_final
    double cone_s   = 0.0;  // with_cone:       the pair + the cone (slot-5 WP-1)
    double iota_s   = 0.0;  // with_sign_split: the involution + the per-sign records
    double presentation_s = 0.0;  // with_presentation (slot-6 WP-2)
    // (slot-7 WP-3, stage 2 item 6) The canonical split of every certified-
    // local ideal of the UNSPLIT record, root finding + rendering, INSIDE
    // run() and inside total_s -- exactly what `--bench ext` timed after run()
    // before item 6, and now reads from here instead of recomputing.  The
    // per-sign splits are inside iota_s.
    double canonical_field_s = 0.0;

    // Inside stage4_s.
    double sturm_bound_s = 0.0;   // B'(k, ell) alone -- the <= 0.1 s gate
    double sturm_sweep_s = 0.0;
    double streamed_s    = 0.0;   // inside sturm_sweep_s
    // (slot-7 WP-3) the cell-local proof_range decision, pool operators
    // included (0 wherever H^0 = 0: the decision is then immediate).
    double proof_range_s = 0.0;
    double finality_s    = 0.0;   // (slot-12 FIN) routes (d0) / (d1), pair build included

    double total_s = 0.0;

    // The gate's three terms.  The denominator is "the wall of the same run
    // WITHOUT the ext layer".
    double pipeline_s() const { return setup_s + stage1_s; }
    double decomp_s() const { return stage2_s; }
    double ext_s() const { return stage3_s + stage4_s + stage5_s + record_s; }
};

// ---------------------------------------------------------------------------
// The driver.
// ---------------------------------------------------------------------------
// (slot-25 PERF-VF (b2)) The G6 Stage-1 release (the family's level (a) + malloc_trim(0)) runs
// only when max_q dim C^q_eq reaches this floor (and HECKE_FIELD_DIET is on).  On the SL(2,Z)
// model max dim C_eq = 9 n + 9 (99 / 1449 / 2709 / 4509 at n = 10 / 160 / 300 / 500), so the floor
// is n >= 299: "below about n = 300 there is no memory to save" (slot24/CLOSE_H2H_diagnosis.md
// Sec. 5.2 (b)).  The census cells (n >= 500) release as before.  Memory / speed only.
constexpr uint32_t kFieldDietReleaseMinDimCeq = 2700;
// (PERF-VF audit S3) Validation hook, calling thread only: the floor in effect is
// kFieldDietReleaseMinDimCeq unless a ScopedReleaseMinDimCeq is alive (0 = release at every
// size, the slot-24 behaviour), so a small cell can exercise the post-release path (anemic pool,
// Sturm sweep reuse, T_p after the release) inside --all.  It never changes a value.
uint32_t field_diet_release_min_dim_ceq();
class ScopedReleaseMinDimCeq {
    uint32_t saved_;
public:
    explicit ScopedReleaseMinDimCeq(uint32_t floor);
    ~ScopedReleaseMinDimCeq();
    ScopedReleaseMinDimCeq(const ScopedReleaseMinDimCeq&) = delete;
    ScopedReleaseMinDimCeq& operator=(const ScopedReleaseMinDimCeq&) = delete;
};
// Process-wide monotone count of the Stage-1 releases run (diagnostic, never a value).
uint64_t stage_one_releases();

class HeckeExtDriver {
public:
    HeckeExtDriver(uint32_t n_sym, uint32_t ell, const ExtDriverOptions& opts = ExtDriverOptions());
    ~HeckeExtDriver();

    HeckeExtDriver(const HeckeExtDriver&) = delete;
    HeckeExtDriver& operator=(const HeckeExtDriver&) = delete;

    // Stages 1-5 and the record.  Exactly once; use resume_u_ell() afterwards.
    void run();

    // [X] Sec. 1.3.1 R3.3 item 4.  See the header comment.  `src` is who is
    // asking for the resume ([KI] Sec. 4d): `Console` (the default, unchanged
    // for every existing caller -- the console `resume` verb and the direct
    // validations) or `Spec` (the campaign, whose resume is spec-driven, not
    // an interactive setting).  It is stamped on the record's own
    // `settings.u_ell.source` exactly where a non-default budget survives to
    // the resumed record -- a resume TO the shipped default budget installs
    // `u_ell = auto`, which IS the default setting, so its source is always
    // `default` whatever `src` is (ScopedUEll's own rule).
    void resume_u_ell(uint32_t budget_mb, hecke::SettingSource src = hecke::SettingSource::Console);

    bool ran() const { return ran_; }

    const HeckeExtRecord&   record() const;
    const ExtDriverTimings& timings() const { return t_; }
    const StageOneOperators& stage_one() const { return s1_; }
    const StreamedOperatorStats& streamed() const { return streamed_; }
    // (slot-7 WP-3) The anemic-pool operators streamed for the cell-local
    // proof_range decision and the atlas label -- kept apart from streamed().
    const StreamedOperatorStats& pool_streamed() const { return pool_streamed_; }
    const std::vector<uint32_t>& H_dimensions() const { return dims_; }
    const std::vector<uint32_t>& degrees_run() const { return degrees_; }
    const std::vector<uint32_t>& u_ell_budget_history() const { return u_ell_history_; }

    // Stage 2's cross-group evaluation test ([X] Sec. 2.2 Stage 2, [XA M5]):
    // how many (later-group block, earlier-group block) pairs carry the SAME
    // maximal ideal.  Recorded as a driver diagnostic; [E] Sec. 7 has no
    // record key for it (WP-2 report finding).
    uint32_t cross_group_matches() const { return cross_group_matches_; }

    // The restricted operators t_p|_V of ideal `i`, for a validation that
    // wants to compare a streamed restriction against a cached one.
    const std::vector<std::pair<uint64_t, ExtMatrix>>& cell_operators(uint32_t degree) const;

    // The Sturm certificate of ideal `i` (empty when (c) did not run).
    const SturmCertificate& sturm_certificate(uint32_t ideal_index) const;

    // The local algebra of ideal `i`.
    const LocalAlgebra& local_algebra(uint32_t ideal_index) const;

    // --- the `hecke_ext/v2` attachments (slot-5 WP-1) ---------------------

    // The cone, when `with_cone` ran; null otherwise.  Its boundary side and
    // its restriction live inside the pair object and outlive it.
    // (slot-12 FIN) The (d1) certificate of the i-th ideal, or nullptr when
    // the route did not run on it (see HeckeFinalityRoutes.h).
    const BoundaryFinalityCertificate* boundary_certificate(uint32_t ideal_index) const;
    const EquivariantMappingCone* cone() const;
    const hecke::SL2ZBoundaryPair* boundary_pair() const;

    // The production involution, when `with_sign_split` ran; null otherwise.
    const hecke::HeckeInvolution* involution() const;
    // The per-degree sign split, indexed as `degrees_run()`; null where the
    // degree carries no cohomology or the split is not applicable.
    const HeckeSignSplit* sign_split(uint32_t degree) const;

    // The two per-sign records of [X] Sec. 6.2 E3 acceptance (c).  Built by
    // `with_sign_split` wherever the split applies; `false` at ell = 2 and
    // where no degree split.  `run()` ASSERTS that the unsplit record's
    // dimensions are the sum of the two.
    bool has_sign_records() const { return sign_records_built_; }
    const HeckeExtRecord& sign_record(bool plus) const;

    // E5-3: the cone ACTION's two RECORDED flags ([C] `sec:recordschema`
    // item 6(e)) and the cross-reference to its own standalone record.  The
    // action is expensive -- its standard-basis sweep is 88 % of its wall --
    // so this driver never builds one; whoever paid for a computed action
    // attaches it here.  Requires a completed run() with `with_cone`.
    void attach_cone_action(const hecke::HeckeConeAction& action);

    // --- the on-request presentation (slot-6 WP-2, OQ11) -------------------
    //
    // Built only under `with_presentation`.  `presentation(i)` is index-
    // aligned with the record's `ideals` (an ideal that is not local has
    // `available = false`); `sign_presentation(plus, i)` with
    // `sign_record(plus).ideals`.  Each object's certificate was checked
    // before it was stored.
    bool has_presentations() const { return presentations_built_; }
    const Presentation& presentation(uint32_t ideal_index) const;
    size_t num_sign_presentations(bool plus) const;
    const Presentation& sign_presentation(bool plus, uint32_t ideal_index) const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;

    void build_stack_();      // setup: K, model, relations, cohomology, family
    // (slot-7 WP-3, stage 2 item 3) The cell-local Dickson-type test of the
    // block `basis_columns` of degree `degree` ([E] sec:schema (a)): true iff
    // its anemic system over AnemicPool(ell) equals none of the cell's H^0
    // systems.  Exact: V (+) H^0 is decomposed jointly over the pool, and a
    // primary component meeting both summands is a shared system.  A joint
    // block that is not certified local and meets both counts as shared, so
    // the test can only err towards `false` (no proof claimed).
    bool cell_local_proof_range_(uint32_t degree, const ExtMatrix& basis_columns);
    // A^{(q)}(T_p) on every degree, p in the anemic pool: the family's for
    // p in S_0, else streamed once and memoised.
    const std::vector<ExtMatrix>& pool_operator_(uint64_t p);
    // (slot-7 WP-3, stage 2 item 5) The atlas label of a local block: its
    // anemic system over the pool, computed AT THE CELL (the block is
    // T-primary once S_sep is in S_0), split in the canonical field, keyed
    // by its least Frobenius conjugate and looked up in the in-process
    // Sigma(ell).  ENGINE_CHECKs that the system is in Sigma ([E] cl:sigma).
    LabelRecord atlas_label_(uint32_t degree, const ExtMatrix& basis_columns);
    // The canonical-split orbit (sorted rows) of a local algebra's own
    // character over its own primes -- how a per-sign ideal is matched to the
    // unsplit ideal whose label it carries.
    std::vector<std::vector<std::vector<uint32_t>>> own_orbit_(const LocalAlgebra& L);
    // (slot-7 WP-3, stage 2 item 6) Attach the `hecke_canonical_split/v1`
    // fragment of a local ideal, through one CanonicalField per (ell, r).
    void attach_split_(IdealRecord& id, const LocalAlgebra& L);
    void stage_one_();
    void stages_two_to_five_();
    void ensure_pair_();         // (slot-12 FIN) the pair for (d1), shared with the cone
    void build_cone_();          // with_cone:       E4b / E5 record objects
    void build_sign_split_();    // with_sign_split: the E3 record objects
    void build_presentations_(); // with_presentation: slot-6 WP-2, no record object
    // (slot-7 WP-2) `presentation = on` (the SETTINGS KEY, read from the
    // record's own settings snapshot): build, then attach one object per
    // ideal of the record and of the per-sign records.
    void presentations_for_settings_();

    uint32_t n_ = 0, ell_ = 0;
    ExtDriverOptions opts_;

    bool ran_ = false;
    StageOneOperators s1_;
    SSepRecord ssep_;          // (slot-7 WP-3, item 4) Stage 2(3) as run
    std::vector<uint32_t> u_ell_history_;
    std::vector<uint32_t> dims_;
    std::vector<uint32_t> degrees_;
    StreamedOperatorStats streamed_;
    StreamedOperatorStats pool_streamed_;   // (slot-7 WP-3) the anemic-pool builds
    ExtDriverTimings t_;
    HeckeExtRecord rec_;
    uint32_t cross_group_matches_ = 0;

    // The `hecke_ext/v2` attachments.  Held here and not in Impl because a
    // caller may read them (the cone is what a validation re-derives against).
    std::unique_ptr<hecke::SL2ZBoundaryPair>   pair_;
    std::unique_ptr<EquivariantMappingCone>    cone_;
    std::unique_ptr<hecke::HeckeInvolution>    involution_;
    // Index-aligned with `degrees_`; a null entry is a degree with no split.
    std::vector<std::unique_ptr<HeckeSignSplit>> splits_;
    bool           sign_records_built_ = false;
    HeckeExtRecord plus_record_, minus_record_;

    // with_presentation (slot-6 WP-2).
    bool presentations_built_ = false;
    std::vector<Presentation> presentations_;
    std::vector<Presentation> sign_presentations_[2];   // [0] plus, [1] minus
};

} // namespace hecke_ext

#endif // HECKE_EXT_DRIVER_H
