// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

#pragma once
#ifndef HECKE_SETTINGS_H
#define HECKE_SETTINGS_H

// ===========================================================================
// (hecke-extensions phase E0b, slot 2 WP-1) THE MATHEMATICAL SETTINGS LAYER
// ([X] Sec. 1.2 `MathSettings`, Sec. 1.5 "The settings layer (R1)", [OD] R1).
// ===========================================================================
//
// Beside the ENGINE knobs (ENGINE_THREADS, ENGINE_DEVICE, ... -- which by
// construction cannot change a value) the pipeline has a second family of
// settings that ARE part of what a result means: how hard the locality loop
// searches, which Sturm cap is in force, whether U_ell is included.  [OD] R1
// makes them ONE struct, owned by the engine:
//
//   * the ENGINE owns the defaults and the validation of values -- the
//     campaign spec and the console only *set* them, by key, through
//     MathSettings::set(); neither re-implements a default or a check;
//   * every record stores the values in force with their `source`;
//   * the FINGERPRINT (FNV-1a-64 over the canonical text of the NON-DEFAULT
//     entries; 0 when everything is at its default) enters every ext hash
//     input and the campaign job id, so a golden mismatch can never be
//     explained away by "a different setting" ([X] Sec. 1.5 rule 1).
//
// SLOT 2 REGISTERED EXACTLY ONE KEY, `locality` ([X] Sec. 1.5 table); SLOT 3
// WP-1 APPENDS `u_ell`.  The remaining ten keys of the table arrive with
// their consuming phases.  Adding a key means, SINCE slot-5 WP-10, exactly
// four edits: a member + its source, a row in registry() (HeckeSettings.cpp),
// an arm in view_of() and a case in set() -- in the registry's fixed key
// order, which is what makes the fingerprint reproducible.  canonical_text(),
// to_json() and text() are loops over that registry built on view_of(), so
// none of them is edited, and a key added to the registry WITHOUT a view_of()
// arm throws rather than emitting an empty value.
//
// STRICT LOCALITY MODE IS IMPLEMENTED since hecke-extensions slot 3 WP-3
// (E0b-S; [X] Sec. 3.1 "Strict mode", [E] Cert. `sec:localcert` "Strict
// mode").  `locality = strict` is a legal value, it round-trips through
// canonical_text(), and it changes the fingerprint; the consumer
// (HeckeModuleDecomposition::compute()) runs the SAME bounded linear witness
// family budget mode runs and then, on a block that family did not decide,
// the terminating Berlekamp route -- so the two modes are BYTE-IDENTICAL
// wherever budget mode certifies, which is every block of the golden grid.
// Strict mode is never the default and never runs on the golden path.
//
// hecke::NotImplementedInThisBuild survives as the named error for the NEXT
// value a build cannot honour; no settings value raises it today.
//
// THREAD-LOCAL, AND PROPAGATED.  The settings in force follow the
// ScopedLegacyHeckeOperatorPath pattern (HeckeOperatorOnK.h:375): a
// thread_local value with a scoped RAII override, registered with
// engine::tape::register_thread_context() so that every task thread
// engine::tape::run_concurrently() starts inherits the launching thread's
// settings.  Two pipelines running concurrently under different `locality`
// therefore cannot see each other's -- and a task thread never silently falls
// back to the defaults.
//
// Build placement: this TU consumes NOTHING of the pipeline (no Matrix, no
// FpPoly, no Hecke object) -- only <string>, ModArithmetic/Fnv.h and the
// engine's thread-context registry -- so it is reachable from anywhere in
// Hecke/ and HeckeExt/ and adds no include-direction obligation.  It lives in
// `Hecke/` and not `HeckeExt/` because its first consumer, `locality`, is
// Hecke/HeckeModuleDecomposition ([X] Sec. 1.1, slot-1 plan Sec. 2.3 item 5:
// the one-way guard HeckeExt -> Hecke must survive).

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace hecke {

// ---------------------------------------------------------------------------
// The named error a settings value that this build cannot honour raises.
// ---------------------------------------------------------------------------
//
// Its `what()` ALWAYS begins with the literal token
// `not_implemented_in_this_build:` so a caller (a campaign driver, a console,
// a validation) can recognise it without depending on the prose.  It is a
// distinct type, never std::invalid_argument: an unparseable value is a user
// error, whereas this is a build-capability statement about a value that IS
// legal.
class NotImplementedInThisBuild : public std::runtime_error {
public:
    explicit NotImplementedInThisBuild(const std::string& detail)
        : std::runtime_error("not_implemented_in_this_build: " + detail) {}
};

// The literal prefix above, for callers that match on it.
extern const char* const kNotImplementedInThisBuildTag;

// ---------------------------------------------------------------------------
// Where a value came from ([X] Sec. 1.5 rule 2: a setting is RECORDED, never
// inferred).
// ---------------------------------------------------------------------------
enum class SettingSource : uint32_t { Default = 0, Spec = 1, Console = 2 };
const char* to_string(SettingSource s);

// ---------------------------------------------------------------------------
// `locality` ([X] Sec. 1.5 table; OQ16, slot-1 confirmation 7).
//   locality = budget=<N>   the default, N = 64 (the shipped
//                           kResidueWitnessMaxAttempts)
//   locality = strict       parsed and validated; NOT IMPLEMENTED in this
//                           build (see the header comment)
// ---------------------------------------------------------------------------
enum class LocalityMode : uint32_t { Budget = 0, Strict = 1 };

// The shipped witness-search budget, and therefore the default.  It caps the
// ONE fixed candidate sequence of [X] Sec. 1.3.1 R3.3 item 2 (the sequence the
// `attempt index` indexes), not a sub-phase of it.
inline constexpr uint32_t kDefaultLocalityBudget = 64;

struct LocalitySetting {
    LocalityMode mode = LocalityMode::Budget;
    // The budget in force.  In strict mode this is the LINEAR phase's budget
    // ([X] Sec. 3.1 step 4: `{mode: strict, budget: N (the linear phase's)}`).
    uint32_t budget = kDefaultLocalityBudget;

    // "budget=<N>" / "strict" -- the campaign-spec / console spelling, and the
    // text the fingerprint is taken over.
    std::string to_text() const;

    bool operator==(const LocalitySetting& o) const {
        return mode == o.mode && budget == o.budget;
    }
    bool operator!=(const LocalitySetting& o) const { return !(*this == o); }
};

// Parses "budget=<N>" (N >= 1) or "strict".  Throws std::invalid_argument with
// a named message on anything else -- including budget=0, which would make the
// loop consume no candidate at all and is a user error, not a mode.
LocalitySetting ParseLocality(const std::string& value);

// ---------------------------------------------------------------------------
// `u_ell` ([X] Sec. 1.5 table, Sec. 2.2 Stage 1, [XA B3]; slot-1 confirmation
// 4; hecke-extensions slot 3 WP-1).
//   u_ell = auto             the default: include U_ell iff the frozen RSS
//                            model of HeckeExt/HeckeUEllBudget.h predicts
//                            <= budget_mb (default 2048) for this (n, ell)
//   u_ell = off              never build U_ell, whatever the prediction
//   u_ell = budget_mb=<M>    auto with the budget pinned to M MB
//
// `auto` and `budget_mb=2048` DECIDE identically -- the decision is a pure
// function of (n, ell, effective_budget_mb()) -- but they are DISTINCT
// settings, because [X] Sec. 1.5 rule 2 is "a setting is recorded, never
// inferred": a user who pinned the budget has that pin in the record and in
// the fingerprint, and a later change of the shipped default cannot silently
// re-interpret their run.  Only `auto` at the shipped budget is the default,
// so only it fingerprints to 0.
//
// THE CONSUMER IS E1b (slot-3 WP-2): HeckeExtDriver's Stage 1.  WP-1 ships
// the key, the pure decision function (HeckeExt/HeckeUEllBudget.h) and their
// validations; NOTHING in this build consults the key, so it cannot move a
// value.
// ---------------------------------------------------------------------------
enum class UEllMode : uint32_t { Auto = 0, Off = 1, BudgetMb = 2 };

// The budget in force at the default ([OD] slot-1 confirmation 4: M = 2048).
inline constexpr uint32_t kDefaultUEllBudgetMb = 2048;

struct UEllSetting {
    UEllMode mode = UEllMode::Auto;
    // Meaningful for Auto and BudgetMb; carried unchanged through Off so the
    // value round-trips and a later `u_ell = auto` finds the shipped default.
    uint32_t budget_mb = kDefaultUEllBudgetMb;

    // The budget the decision function must be called with.  `off` has none:
    // its verdict is `off`, reached without forming a prediction.
    uint32_t effective_budget_mb() const { return budget_mb; }

    // "auto" / "off" / "budget_mb=<M>" -- the campaign-spec / console
    // spelling, and the text the fingerprint is taken over.
    std::string to_text() const;

    bool operator==(const UEllSetting& o) const {
        return mode == o.mode && budget_mb == o.budget_mb;
    }
    bool operator!=(const UEllSetting& o) const { return !(*this == o); }
};

// Parses "auto", "off" or "budget_mb=<M>" (M >= 1).  Throws
// std::invalid_argument with a named message on anything else -- including
// budget_mb=0, which would refuse every cell and is a user error, not a mode
// (`off` is the mode that refuses every cell).
UEllSetting ParseUEll(const std::string& value);

// ---------------------------------------------------------------------------
// `sturm` / `sturm_unavailable` ([X] Sec. 1.5 table, rows `sturm.cap` and
// `sturm.unavailable`; OQ15, slot-1 confirmation 6; hecke-extensions slot 3
// WP-2, whose HeckeExt/HeckeSturmSweep IS the consumer).
//
//   sturm = auto             the default: the bound search stops at the cap
//                            12*B(k), B(k) = floor(k/12) + 1
//   sturm = off              certificate (c) is not attempted at all
//   sturm = cap=<B>          auto with the cap pinned to B (B >= 2)
//
//   sturm_unavailable = nonfinal    the default: a `none_at_cap` bound leaves
//                                   the ideal `final = false` with reason
//                                   `sturm_unavailable`, and NOTHING is swept
//   sturm_unavailable = heuristic   the sweep runs over the primes <= cap
//                                   anyway and the ideal is `final =
//                                   "heuristic"` with `sturm_primes`
//
// The KEY NAMES here are the campaign-spec / console SPELLINGS of [X]
// Sec. 1.5 (`sturm`, `sturm_unavailable`); the settings they carry are that
// table's `sturm.cap` and `sturm.unavailable`, and the record keys they feed
// are `sturm.cap` and `sturm.mode`.  Only the defaults fingerprint to 0.
//
// The cap is stored as a MODE plus a number and never as a resolved integer,
// because 12*B(k) depends on the weight k = n + 2, which this dependency-free
// TU does not know.  The consumer resolves it
// (HeckeExt/HeckeSturmSweep.h: SturmSweepOptions::from_settings(s, k), which
// calls HeckeExt/HeckeSturmBound's default_cap(k)).
// ---------------------------------------------------------------------------
enum class SturmCapMode : uint32_t { Auto = 0, Off = 1, Cap = 2 };

struct SturmSetting {
    SturmCapMode mode = SturmCapMode::Auto;
    // Meaningful iff mode == Cap; carried unchanged otherwise so the value
    // round-trips.
    uint32_t cap = 0;

    // "auto" / "off" / "cap=<B>" -- the spelling, and the text the
    // fingerprint is taken over.
    std::string to_text() const;

    bool operator==(const SturmSetting& o) const {
        return mode == o.mode && cap == o.cap;
    }
    bool operator!=(const SturmSetting& o) const { return !(*this == o); }
};

// Parses "auto", "off" or "cap=<B>" (B >= 2, the least prime the bound search
// can stop at).  Throws std::invalid_argument with a named message otherwise.
SturmSetting ParseSturm(const std::string& value);

enum class SturmUnavailablePolicy : uint32_t { NonFinal = 0, Heuristic = 1 };

// "nonfinal" / "heuristic".
const char* to_text(SturmUnavailablePolicy p);
// Parses exactly those two; throws std::invalid_argument otherwise.
SturmUnavailablePolicy ParseSturmUnavailable(const std::string& value);

// ---------------------------------------------------------------------------
// `commutant_budget` ([X] Sec. 1.5's "the commutant's kCommutantBudget (a
// named constant today, a setting only if a consumer asks)"; [OD] slot-3
// confirmation 3 lists "kCommutantBudget as a setting" as E2 content, and the
// owner's later text wins -- the CONSUMER EXISTS: [X] Sec. 1.3.1 R3.1 item 4
// makes a resume with a changed budget A NEW RECORDED SETTING, and E2's
// record is what carries `resumable.budget`.  hecke-extensions slot 4, WP-1.)
//
//   commutant_budget = <N>     N modmul; the default is 2*10^10
//
// THE DEFAULT LIVES HERE, not in HeckeExt/, because `Hecke/` may not include
// `HeckeExt/` (check_heckeext_includes.sh rule 1) -- the `u_ell` precedent
// inverted.  HeckeExt/HeckeGenerationCertificates' CommutantRoute::
// kCommutantBudget is an ALIAS of this constant, so there is ONE definition
// and not a second literal.
// ---------------------------------------------------------------------------

// [X] Sec. 2.2 Stage 4(b) step 2: "attempted only if k*D^2*u^2 <=
// kCommutantBudget = 2*10^10 modmul".
inline constexpr uint64_t kDefaultCommutantBudget = 20000000000ull;  // 2 * 10^10

struct CommutantBudgetSetting {
    uint64_t budget = kDefaultCommutantBudget;

    // "<N>" -- the campaign-spec / console spelling, and the text the
    // fingerprint is taken over.
    std::string to_text() const;

    bool operator==(const CommutantBudgetSetting& o) const { return budget == o.budget; }
    bool operator!=(const CommutantBudgetSetting& o) const { return !(*this == o); }
};

// Parses a decimal N >= 1.  Throws std::invalid_argument otherwise -- N = 0
// would refuse every commutant system before its first row, which is a user
// error and not a mode (`over_budget` is what a real budget produces).
CommutantBudgetSetting ParseCommutantBudget(const std::string& value);

// ---------------------------------------------------------------------------
// `final_mode` ([X] Sec. 1.5 table row 1, Sec. 1.3.1 R3.2; OQ8, slot-1
// confirmation 8).  hecke-extensions slot 4, WP-1 -- its consumer, E2's
// `hecke_ext::is_final`, ships in the same package.
//
//   final_mode = closure           the DEFAULT: an ideal is `final = true`
//                                  only on a PROOF ((a), (b), or (c) on its
//                                  proof range)
//   final_mode = sturm_heuristic   the user-selected mode of [E]
//                                  `cert:sturm`'s HEURISTIC RANGE: a complete
//                                  sweep off the proof range is
//                                  `final = "heuristic"` instead of `false`
//
// The RECORD-LAYER twin of this enum is hecke_ext::FinalMode
// (HeckeExt/HeckeExtFinal.h); this TU may not include HeckeExt/, so the two
// are separate types kept in step by HeckeExtRecordValidation.
// ---------------------------------------------------------------------------
enum class FinalMode : uint32_t { Closure = 0, SturmHeuristic = 1 };

// "closure" / "sturm_heuristic".
const char* to_text(FinalMode m);
// Parses exactly those two; throws std::invalid_argument otherwise.
FinalMode ParseFinalMode(const std::string& value);

// ---------------------------------------------------------------------------
// `field_polynomial` ([X] Sec. 1.5 table, Sec. 2.2 Stage 6; [OD] OQ9; [E]
// `def:canon`).  hecke-extensions slot 5, WP-7 (phase E6) -- the consumer is
// HeckeExt/HeckeCanonicalField's FieldPolynomialFromSettings().
//
//   field_polynomial = least     the DEFAULT: C_{ell,r} is the
//                                CanonicalLess-least monic irreducible of
//                                degree r over F_ell ([E] `def:canon`), found
//                                by enumeration in that order and certified
//                                with hecke::CertifyIrreducible
//   field_polynomial = conway    RESERVED.  Parsed, validated and round-tripped
//                                -- it is a LEGAL value -- but no build ships
//                                Conway tables, so the consumer raises
//                                hecke::NotImplementedInThisBuild.  It is
//                                never silently degraded to `least`, because
//                                the defining polynomial IS part of what the
//                                reported eigenvalues mean.
//
// The `locality = strict` precedent inverted in time: there the value was
// parsed before its implementation existed and the consumer raised until slot
// 3; here the value is parsed and the consumer raises until Conway tables
// exist ([X] Sec. 6.5, "a follow-up tied to that key").
// ---------------------------------------------------------------------------
enum class FieldPolynomialMode : uint32_t { Least = 0, Conway = 1 };

// "least" / "conway".
const char* to_text(FieldPolynomialMode m);
// Parses exactly those two; throws std::invalid_argument otherwise.
FieldPolynomialMode ParseFieldPolynomial(const std::string& value);

// ---------------------------------------------------------------------------
// `split_field_report` ([X] Sec. 1.5 table, Sec. 2.2 Stage 6; [OD] OQ10; [E]
// `sec:field`).  hecke-extensions slot 5, WP-7 (phase E6).
//
//   split_field_report = canonical   the DEFAULT since [E] v1.4.0 (OQ10): the
//                                    eigenvalues are re-embedded into
//                                    F_ell[Y]/(C_{ell,r}) and reported there,
//                                    so two runs and two blocks are comparable
//   split_field_report = own_field   the secondary form, the block's own field
//                                    F_ell[X]/(pi_w) with the P_p
//
// NEITHER FORM IS EVER LOST: tier (i) is "always emitted as
// coefficient_field" whichever way the key points ([E] `sec:field`), so
// the key selects the PRIMARY form only.
//
// NOTHING IN THIS BUILD CONSULTS THE KEY.  The report consumer is E6's, in
// slot 6 -- the `u_ell` precedent of slot-3 WP-1, where the key, its parser
// and its validation shipped one slot before Stage 1 read it.  The default is
// ALSO conditional on E6's performance gate, RESTATED by [OD] E6-1
// (2026-09-22, `docs/implementation/e6_gate_brief.md`): `canonical_field_s <=
// 0.10 * total_s`, `total_s` being the wall of `HeckeExtDriver::run()`, not
// `ext_s` -- the pre-slot-6 `<= 2 * own_field_s` half is deleted, not
// meetable by any admissible implementation.  Slot-5 WP-7 and slot-6 WP-7
// MEASURE that gate through `--bench ext` and deliberately do NOT flip the
// default on it; E6 acceptance against the restated gate is slot 7's.
// ---------------------------------------------------------------------------
enum class SplitFieldReport : uint32_t { Canonical = 0, OwnField = 1 };

// "canonical" / "own_field".
const char* to_text(SplitFieldReport m);
// Parses exactly those two; throws std::invalid_argument otherwise.
SplitFieldReport ParseSplitFieldReport(const std::string& value);

// ---------------------------------------------------------------------------
// `presentation` ([X] Sec. 1.5 table, Sec. 2.2 Stage 5; [OD] OQ11; [E]
// `sec:schema`, slot-7 owner confirmation 15).  hecke-extensions slot 7,
// WP-2 ([X] Sec. 8 stage 1 (2)).
//
//   presentation = off   the DEFAULT: no `presentation` object in the record
//   presentation = on    every ideal entry of the record (and of the per-sign
//                        records) carries `presentation` =
//                        {"generator_basis":"block","relation_basis":
//                        "algebra_rref","fragment":<hecke_presentation/v1>}
//
// The consumer is HeckeExt/HeckeExtDriver (the key switches the on-request
// presentation on AND attaches it to the record; the pre-existing
// `ExtDriverOptions::with_presentation` still only computes).
// ---------------------------------------------------------------------------
enum class PresentationMode : uint32_t { Off = 0, On = 1 };

// "off" / "on".
const char* to_text(PresentationMode m);
// Parses exactly those two; throws std::invalid_argument otherwise.
PresentationMode ParsePresentation(const std::string& value);

// ---------------------------------------------------------------------------
// `predictions` ([X] Sec. 1.5 table; [OD] OQ14).  hecke-extensions slot 7,
// WP-2 ([X] Sec. 8 stage 1 (2)).
//
//   predictions = annotate   the DEFAULT: predictions are annotations
//                            (predicted / confirmed / unpredicted)
//   predictions = assert     the debug switch: a refuted prediction fails the
//                            run -- used by validations, never by a campaign
//
// NOTHING IN THIS BUILD CONSULTS THE KEY: its consumer (the record's
// `predictions_mode`) is slot-7 WP-3 item 7 -- the `split_field_report`
// precedent of slot 5 (key, parser and validation one package ahead of the
// reader).
// ---------------------------------------------------------------------------
enum class PredictionsMode : uint32_t { Annotate = 0, Assert = 1 };

// "annotate" / "assert".
const char* to_text(PredictionsMode m);
// Parses exactly those two; throws std::invalid_argument otherwise.
PredictionsMode ParsePredictions(const std::string& value);

// ---------------------------------------------------------------------------
// `coefficient_ring` ([E] v1.9.0-pre.1 sec:ring (a); [X] Sec. 1.5; slot-8
// decision 4 (A)).  hecke-extensions slot 8, WP-3 (E8 item 3).
//
//   coefficient_ring = F_ell        the DEFAULT: coefficients in the field F_ell
//   coefficient_ring = Z/ell^<k>    2 <= k <= 4096 (slot-11 WP-18c; 63 through
//                                   [E] v1.12.0-pre.1), k in decimal without sign,
//                                   leading zero or space; the token `ell` is
//                                   LITERAL -- the setting is run-level and the
//                                   prime is the cell's (Z/ell^2 at (n, 5) is Z/25)
//
//   coefficient_ring = Z            (slot-24 D6, DESCENT_plan.md Sec. 4 D6,
//                                   OD-10 (a); UN-RESERVED) integral
//                                   coefficients: stored as k = 0, honoured by
//                                   the integral descent ONLY (Integral/Descent/
//                                   DescentRecord.h; [C] thm:descentcert);
//                                   every reader of k refuses it by name
//                                   (CoefficientRingK(who), exit 5)
//
// RESERVED (legal, a build-capability statement): (until slot 24 D6 also `Z`)
// and `Z/<m>` with a literal integer m >= 2 (a general Z/m, outside
// ENGINE_API.md Sec. 13).  ParseCoefficientRing -- and hence
// MathSettings::set() -- REFUSES them ITSELF by raising
// hecke::NotImplementedInThisBuild naming the key and the value (console exit
// 5): unlike `conway`, a reserved ring is never stored, because a ring in
// force reaches every construction site of the pipeline ([E] sec:ring (a),
// "Reserved values").  Every other text -- `Z/ell^1`, `Z/ell^0`, `Z/ell^4097`,
// `Z/ell^02`, `F_5`, ... -- is std::invalid_argument.
//
// The ring is read ONCE, at construction, by the objects that honour it
// (EquivariantCohomologyOnK, HeckeOperatorOnK; Hecke/CoefficientRing.h), and
// every other construction site or field-only consumer REFUSES a non-default
// value by hecke::NotImplementedInThisBuild naming itself ([E] sec:ring (b)).
// ---------------------------------------------------------------------------
struct CoefficientRingSetting {
    uint32_t k = 1;   // 1 = F_ell (the default); 2 .. 4096 = Z/ell^k; (slot-24 D6) 0 = Z, the integral descent

    bool is_field() const { return k == 1; }
    bool is_integral() const { return k == 0; }   // (slot-24 D6) `coefficient_ring = Z`
    // "F_ell" / "Z/ell^<k>" -- the canonical text ([E] sec:ring (a)).
    std::string to_text() const;

    bool operator==(const CoefficientRingSetting& o) const { return k == o.k; }
    bool operator!=(const CoefficientRingSetting& o) const { return !(*this == o); }
};

// The grammar above: F_ell / Z/ell^<k> parsed; Z and Z/<m> raise
// NotImplementedInThisBuild; everything else std::invalid_argument.
CoefficientRingSetting ParseCoefficientRing(const std::string& value);

// ---------------------------------------------------------------------------
// The settings in force.
// ---------------------------------------------------------------------------
struct MathSettings {
    LocalitySetting locality;
    SettingSource   locality_source = SettingSource::Default;
    UEllSetting     u_ell;
    SettingSource   u_ell_source = SettingSource::Default;
    // (slot-3 WP-2 / E1b) APPENDED after `u_ell`, never inserted.
    SturmSetting    sturm;
    SettingSource   sturm_source = SettingSource::Default;
    SturmUnavailablePolicy sturm_unavailable = SturmUnavailablePolicy::NonFinal;
    SettingSource   sturm_unavailable_source = SettingSource::Default;
    // (slot-4 WP-1 / E2a) APPENDED after `sturm_unavailable`, never inserted
    // -- the fingerprint is a function of the registry ORDER, and both keys
    // are at their defaults, so the all-defaults fingerprint is still 0 and no
    // published job id moved.
    CommutantBudgetSetting commutant_budget;
    SettingSource   commutant_budget_source = SettingSource::Default;
    FinalMode       final_mode = FinalMode::Closure;
    SettingSource   final_mode_source = SettingSource::Default;
    // (slot-5 WP-7 / E6) APPENDED after `final_mode`, never inserted.  Both
    // are at their defaults, so the all-defaults fingerprint is still 0 and no
    // published job id, record hash or golden value moved.
    FieldPolynomialMode field_polynomial = FieldPolynomialMode::Least;
    SettingSource   field_polynomial_source = SettingSource::Default;
    SplitFieldReport split_field_report = SplitFieldReport::Canonical;
    SettingSource   split_field_report_source = SettingSource::Default;
    // (slot-7 WP-2, [X] Sec. 8 stage 1 (2)) APPENDED after
    // `split_field_report`, never inserted.  Both at their defaults, so the
    // all-defaults fingerprint is still 0; and since stage 1 (1) the golden
    // record texts see `settings` only as its fingerprint, so the
    // registration moved no golden line.
    PresentationMode presentation = PresentationMode::Off;
    SettingSource   presentation_source = SettingSource::Default;
    PredictionsMode predictions = PredictionsMode::Annotate;
    SettingSource   predictions_source = SettingSource::Default;
    // (slot-8 WP-3, E8 item 3) Appended at the registry's tail: eleven keys.
    // At its default the key contributes nothing to canonical_text(), so the
    // all-defaults fingerprint is still 0 and the registration is golden-
    // additive (the slot-7 stage 1 (1) recipe; [E] sec:ring (a)).
    CoefficientRingSetting coefficient_ring;
    SettingSource   coefficient_ring_source = SettingSource::Default;

    // The all-defaults instance ([X] Sec. 1.5 rule 1: this is what the golden
    // harness, --bench and every golden-comparing validation arm run under).
    static const MathSettings& defaults();

    bool is_default() const;

    // The canonical text of the NON-DEFAULT entries, in the registry's fixed
    // key order, one `key = value` per line, '\n'-terminated.  Empty when
    // everything is at its default.
    std::string canonical_text() const;

    // FNV-1a-64 (src/ModArithmetic/Fnv.h, the ONE byte-wise FNV)
    // over canonical_text(); 0 when everything is at its default.
    uint64_t fingerprint() const;

    // The record's `settings` block: every key with its value and its source,
    // plus the fingerprint.  Fixed key order, no hash containers.
    std::string to_json() const;

    // Sets one key by its campaign-spec / console spelling.  Returns false and
    // leaves *this untouched for an UNKNOWN key (the caller reports a spec
    // error, CampaignSpec.cpp:304's convention); throws std::invalid_argument
    // for a known key with a bad value.  This is the ONE validation point
    // ([OD] R1: "never twice").
    bool set(const std::string& key, const std::string& value, SettingSource src);

    // The canonical TEXT of one key's current value -- `set()`'s inverse, and
    // the one place a key's spelling is produced (slot-4 WP-1/WP-5 integration).
    //
    // WHY IT EXISTS.  A consumer that must render or seed every key generically
    // -- the console's `InstallMathSettings`, which walks `keys()` -- used to
    // carry a hand-written per-key switch of its own, which silently returned
    // the empty string for any key appended after it was written ([OD] R1: "the
    // engine owns the defaults ... the console never re-implements them").  The
    // console asking the engine is the whole point; a second copy of the key
    // list is the defect class.
    //
    // For every key of `keys()` this returns EXACTLY the text `canonical_text()`
    // and `to_json()` emit for it, so a round trip `set(k, text(k), src)` is the
    // identity on the value.
    //
    // SINCE hecke-extensions slot 5 (WP-10 item 2) THAT IS TRUE BY
    // CONSTRUCTION, not by repetition: `canonical_text()`, `to_json()` and this
    // accessor are all loops over `keys()` built on ONE per-key view
    // (HeckeSettings.cpp's `view_of`), so a key is spelled in exactly one
    // place.  [S4] WP-1 had deliberately left the other two alone, on the
    // grounds that they are the fingerprint's and the record's basis; WP-10
    // takes the refactor under the gate that makes it safe -- the emitted text
    // is asserted byte-unchanged on every key and on a non-default combination
    // (WP-10 acceptance (c)), and the all-defaults fingerprint is still 0.
    //
    // THROWS `std::invalid_argument` for an unknown key -- and, deliberately,
    // for a key that is registered in `keys()` but not handled by `view_of`, so
    // the next key appended to the registry fails LOUDLY in the first console
    // session -- and now in `canonical_text()` and `to_json()` too -- instead
    // of silently seeding "" or being dropped from the fingerprint.
    std::string text(const std::string& key) const;

    // The registered keys, in the fixed order canonical_text() uses.
    static const std::vector<std::string>& keys();
    static bool known_key(const std::string& key);

    bool operator==(const MathSettings& o) const {
        return locality == o.locality && locality_source == o.locality_source &&
               u_ell == o.u_ell && u_ell_source == o.u_ell_source &&
               sturm == o.sturm && sturm_source == o.sturm_source &&
               sturm_unavailable == o.sturm_unavailable &&
               sturm_unavailable_source == o.sturm_unavailable_source &&
               commutant_budget == o.commutant_budget &&
               commutant_budget_source == o.commutant_budget_source &&
               final_mode == o.final_mode && final_mode_source == o.final_mode_source &&
               field_polynomial == o.field_polynomial &&
               field_polynomial_source == o.field_polynomial_source &&
               split_field_report == o.split_field_report &&
               split_field_report_source == o.split_field_report_source &&
               presentation == o.presentation && presentation_source == o.presentation_source &&
               predictions == o.predictions && predictions_source == o.predictions_source &&
               coefficient_ring == o.coefficient_ring &&
               coefficient_ring_source == o.coefficient_ring_source;
    }

    // (slot-7 WP-2) The same values, every NON-default key re-stamped with
    // `src` and every default key with `Default`.  Generic over keys(): it is
    // set(key, text(key), ...) per key, so it needs no edit when a key is
    // appended.  What ScopedMathSettings(s, src) installs.
    MathSettings with_source(SettingSource src) const;
    bool operator!=(const MathSettings& o) const { return !(*this == o); }
};

// ---------------------------------------------------------------------------
// The thread-local settings in force, and the scoped override.
// ---------------------------------------------------------------------------
const MathSettings& math_settings();
void set_math_settings(const MathSettings& s);

class ScopedMathSettings {
    MathSettings saved_;
public:
    // Installs `s` with the sources it carries.
    explicit ScopedMathSettings(const MathSettings& s) : saved_(math_settings()) {
        set_math_settings(s);
    }
    // (slot-7 WP-2, [X] Sec. 1.5 rule 2) Installs `s` with every non-default
    // key stamped `src` (and every default key `default`): the caller says
    // who set the values -- `spec` for a campaign spec or the golden harness
    // (slot-7 owner confirmation 13), `console` for a console session.
    ScopedMathSettings(const MathSettings& s, SettingSource src) : saved_(math_settings()) {
        set_math_settings(s.with_source(src));
    }
    ~ScopedMathSettings() { set_math_settings(saved_); }
    ScopedMathSettings(const ScopedMathSettings&) = delete;
    ScopedMathSettings& operator=(const ScopedMathSettings&) = delete;
};

// Convenience: override `u_ell` alone, leaving every other key where it is.
// (slot-7 WP-2) `src` is the source a NON-default value is stamped with; it
// defaults to `console`, which is what every pre-slot-7 caller got, so those
// callers are unchanged.  The golden harness passes `spec` (slot-7 owner
// confirmation 13; [X] Sec. 1.5 `source in {default, spec, console}`).  A
// value at its default is stamped `default` whatever `src` says.
class ScopedUEll {
    MathSettings saved_;
public:
    explicit ScopedUEll(const UEllSetting& u, SettingSource src = SettingSource::Console)
        : saved_(math_settings()) {
        MathSettings s = saved_;
        s.u_ell = u;
        s.u_ell_source = (u == UEllSetting()) ? SettingSource::Default : src;
        set_math_settings(s);
    }
    ~ScopedUEll() { set_math_settings(saved_); }
    ScopedUEll(const ScopedUEll&) = delete;
    ScopedUEll& operator=(const ScopedUEll&) = delete;
};

// Convenience: override `locality` alone, leaving every other key where it is.
class ScopedLocality {
    MathSettings saved_;
public:
    explicit ScopedLocality(const LocalitySetting& l) : saved_(math_settings()) {
        MathSettings s = saved_;
        s.locality = l;
        s.locality_source = (l == LocalitySetting()) ? SettingSource::Default
                                                     : SettingSource::Console;
        set_math_settings(s);
    }
    ScopedLocality(LocalityMode mode, uint32_t budget)
        : ScopedLocality(LocalitySetting{mode, budget}) {}
    ~ScopedLocality() { set_math_settings(saved_); }
    ScopedLocality(const ScopedLocality&) = delete;
    ScopedLocality& operator=(const ScopedLocality&) = delete;
};

// Convenience: override the two `sturm.*` keys alone (slot-3 WP-2).
class ScopedSturm {
    MathSettings saved_;
public:
    ScopedSturm(const SturmSetting& cap, SturmUnavailablePolicy policy)
        : saved_(math_settings()) {
        MathSettings s = saved_;
        s.sturm = cap;
        s.sturm_source = (cap == SturmSetting()) ? SettingSource::Default
                                                 : SettingSource::Console;
        s.sturm_unavailable = policy;
        s.sturm_unavailable_source = (policy == SturmUnavailablePolicy::NonFinal)
                                         ? SettingSource::Default
                                         : SettingSource::Console;
        set_math_settings(s);
    }
    ~ScopedSturm() { set_math_settings(saved_); }
    ScopedSturm(const ScopedSturm&) = delete;
    ScopedSturm& operator=(const ScopedSturm&) = delete;
};

// Convenience: override `commutant_budget` alone (slot-4 WP-1).  The
// CheckThread arm of WP-1 runs two run_concurrently() tasks under two
// different budgets through this guard.
class ScopedCommutantBudget {
    MathSettings saved_;
public:
    explicit ScopedCommutantBudget(uint64_t budget) : saved_(math_settings()) {
        MathSettings s = saved_;
        s.commutant_budget.budget = budget;
        s.commutant_budget_source = (budget == kDefaultCommutantBudget)
                                        ? SettingSource::Default
                                        : SettingSource::Console;
        set_math_settings(s);
    }
    ~ScopedCommutantBudget() { set_math_settings(saved_); }
    ScopedCommutantBudget(const ScopedCommutantBudget&) = delete;
    ScopedCommutantBudget& operator=(const ScopedCommutantBudget&) = delete;
};

// Convenience: override the two Stage-6 report keys alone (slot-5 WP-7).
// (slot-8 WP-3) `coefficient_ring` alone: Z/ell^k for k >= 2, F_ell for
// k == 1 (source Default then).  The golden harness's E8 points (WP-4) and
// HeckeReductionValidation install the ring through this, around the E8
// computations only ([E] sec:ring (e)).
class ScopedCoefficientRing {
    MathSettings saved_;
public:
    explicit ScopedCoefficientRing(uint32_t k, SettingSource src = SettingSource::Spec)
        : saved_(math_settings()) {
        MathSettings s = saved_;
        s.coefficient_ring.k = k;
        s.coefficient_ring_source = (k == 1) ? SettingSource::Default : src;
        set_math_settings(s);
    }
    ~ScopedCoefficientRing() { set_math_settings(saved_); }
    ScopedCoefficientRing(const ScopedCoefficientRing&) = delete;
    ScopedCoefficientRing& operator=(const ScopedCoefficientRing&) = delete;
};

// ---------------------------------------------------------------------------
// (slot-8 WP-3) THE REFUSAL RULE'S GUARD ([E] sec:ring (b); Hecke/
// CoefficientRing.h for the honouring side).
// ---------------------------------------------------------------------------
// The exponent k of the ring in force on this thread (1 = F_ell, the default).
// (slot-24 D6) Under `coefficient_ring = Z` (stored as k = 0) it REFUSES
// (NotImplementedInThisBuild naming `who`, console exit 5): every reader of k
// is a field / Z/ell^k construction site, and Z is honoured by the integral
// descent alone (CoefficientRingIsZ(); Integral/Descent/DescentRecord.h).
uint32_t CoefficientRingK();
uint32_t CoefficientRingK(const std::string& who);
// (slot-24 D6) True iff `coefficient_ring = Z` is in force on this thread.
bool CoefficientRingIsZ();
// "F_ell" / "Z/ell^<k>" -- the canonical text of [E] sec:ring (a).
std::string CoefficientRingText(uint32_t k);
// Raises NotImplementedInThisBuild("<who>: coefficient_ring=Z/ell^<k> is not
// honoured in this build<why>").
[[noreturn]] void RefuseCoefficientRing(const std::string& who, uint32_t k, const std::string& why = "");
// The constructor-entry guard of a REFUSING member ([E] sec:ring (b2)):
// refuses when the ring in force is not F_ell, or when an input object was
// built over Z/ell^{input_k} (input_k >= 2).  A no-op at the default with
// field inputs -- the only case any existing path reaches.
void RequireFieldCoefficientRing(const std::string& who, uint32_t input_k = 1);

class ScopedSplitField {
    MathSettings saved_;
public:
    ScopedSplitField(FieldPolynomialMode fp, SplitFieldReport sfr) : saved_(math_settings()) {
        MathSettings s = saved_;
        s.field_polynomial = fp;
        s.field_polynomial_source = (fp == FieldPolynomialMode::Least)
                                        ? SettingSource::Default
                                        : SettingSource::Console;
        s.split_field_report = sfr;
        s.split_field_report_source = (sfr == SplitFieldReport::Canonical)
                                          ? SettingSource::Default
                                          : SettingSource::Console;
        set_math_settings(s);
    }
    ~ScopedSplitField() { set_math_settings(saved_); }
    ScopedSplitField(const ScopedSplitField&) = delete;
    ScopedSplitField& operator=(const ScopedSplitField&) = delete;
};

} // namespace hecke

#endif // HECKE_SETTINGS_H
