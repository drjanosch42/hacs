// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// hecke-extensions phase E2a (slot 4, WP-1) -- HeckeExtRecordValidation.
//
// See HeckeExtRecordValidation.h.  assert() is not used: every check is
// counted and reported and the return value is the number of failures, so a
// Release build reports rather than aborts (the HeckeExtAlgebraValidation /
// HeckeExtSturmSweepValidation idiom).

#include "HeckeExtRecordValidation.h"

#include "HeckeAtlas.h"
#include "HeckeExtComplexRecords.h"
#include "HeckeExtDriver.h"
#include "HeckeExtFinal.h"
#include "HeckeExtRecord.h"
#include "HeckeExtReport.h"
#include "HeckeFinalityRoutes.h"
#include "HeckeGenerationCertificates.h"
#include "HeckeLocalAlgebra.h"
#include "HeckeModuleStructure.h"
#include "HeckeSignSplit.h"
#include "HeckeSturmSweep.h"
#include "HeckeUEllBudget.h"

#include "../Campaign/CampaignDriver.h"
#include "../Campaign/CampaignSpec.h"
#include "../Hecke/HeckeModuleDecomposition.h"
#include "../Hecke/HeckeOperatorFamily.h"
#include "../Hecke/HeckeSettings.h"
#include "../Hecke/SL2ZTessellationTables.h"
#include "../EquivariantCohomologyOnK.h"
#include "../LocalSystemSL2Z.h"
#include "../LocalSystemSimplicialComplex.h"

#include "../../ModArithmetic/Fnv.h"
#include "../../ModArithmetic/Matrix.h"
#include "../Util/ValidationCheck.h"

#include <unistd.h>

#include <algorithm>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <thread>
#include <sstream>
#include <string>
#include <type_traits>
#include <vector>

namespace {
// Termux has no writable /tmp: honour $TMPDIR (std::filesystem::temp_directory_path), falling back to /tmp.
std::string TmpDir() {
    std::error_code ec;
    const auto p = std::filesystem::temp_directory_path(ec);
    return ec ? std::string("/tmp") : p.string();
}


using hecke_ext::CommutantOutcome;
using hecke_ext::CommutantRoute;
using hecke_ext::ExtMatrix;
using hecke_ext::Final;
using hecke_ext::FinalInputs;
using hecke_ext::Finality;
using hecke_ext::HeckeExtRecord;
using hecke_ext::IdealRecord;
using hecke_ext::LocalAlgebra;
using hecke_ext::ModuleStructure;
using hecke_ext::ResumableRecord;
using hecke_ext::SturmSweepOutcome;

oc_util::Checker Check("  [FAIL] ");

template <typename T>
void CheckEq(const T& got, const T& want, const std::string& what) {
    ++Check.checks;
    if (!(got == want)) {
        ++Check.fails;
        std::cout << "  [FAIL] " << what << ": got " << got << ", want " << want << std::endl;
    }
}

void CheckVec(const std::vector<uint32_t>& got, const std::vector<uint32_t>& want,
              const std::string& what) {
    ++Check.checks;
    if (got != want) {
        ++Check.fails;
        std::ostringstream oss;
        oss << "  [FAIL] " << what << ": got [";
        for (size_t i = 0; i < got.size(); ++i) { if (i) oss << ","; oss << got[i]; }
        oss << "], want [";
        for (size_t i = 0; i < want.size(); ++i) { if (i) oss << ","; oss << want[i]; }
        oss << "]";
        std::cout << oss.str() << std::endl;
    }
}

bool HasKey(const std::string& json, const std::string& key) {
    return json.find("\"" + key + "\":") != std::string::npos;
}

std::string HexU64(uint64_t v) {
    static const char* kHex = "0123456789abcdef";
    std::string out(16, '0');
    for (int i = 0; i < 16; ++i) out[15 - i] = kHex[(v >> (4 * i)) & 0xfu];
    return out;
}

// ---------------------------------------------------------------------------
// The golden COLLECTORS of [X] Sec. 1.6, defined and called HERE ONLY.
//
// Slot 4 emits no golden entry (slot-4 plan Sec. 1, exit item 1), so these
// must not be reachable from Golden/GoldenHarness.cpp's RunCell32 / RunCell64.
// They live in this TU precisely so that they cannot be: HeckeExt/ headers are
// not included by the harness, and these three functions are file-local.
// ---------------------------------------------------------------------------
uint64_t CollectExtIdeals(const HeckeExtRecord& r) {
    uint64_t h = engine::kFnvOffset64;
    for (const IdealRecord& id : r.ideals) {
        h = engine::FnvU64(h, id.id);
        h = engine::FnvU64(h, id.dim);
        h = engine::FnvU64(h, id.residue_degree);
        h = engine::FnvU64(h, id.local ? 1u : 0u);
        h = engine::FnvU64(h, static_cast<uint64_t>(id.final.value));
        h = engine::FnvBytes(h, id.final.reason);
    }
    return h;
}

uint64_t CollectExtCertify(const HeckeExtRecord& r) {
    uint64_t h = engine::FnvU64(engine::kFnvOffset64, r.verify());
    for (const IdealRecord& id : r.ideals) {
        h = engine::FnvBytes(h, id.certificate);
        h = engine::FnvBytes(h, id.certificate_routes.a);
        h = engine::FnvBytes(h, id.certificate_routes.b);
        h = engine::FnvBytes(h, id.certificate_routes.c);
        for (const hecke_ext::ModuleRecordEntry& m : id.modules) {
            h = engine::FnvU64(h, m.structure.verify());
        }
    }
    return h;
}

uint64_t CollectExtJsonCanonical(const HeckeExtRecord& r) {
    return engine::FnvBytes(engine::kFnvOffset64, hecke_ext::BuildHeckeExtReportJson(r));
}

// ---------------------------------------------------------------------------
// The pipeline stack -- the HeckeExtSturmSweepValidation pattern.
// ---------------------------------------------------------------------------
struct Stack {
    SimplicialComplex K;
    hecke::SL2ZTessellationTables model;
    LocalSystemSimplicialComplex lssc;
    EquivariantCohomologyOnK coh;

    Stack(uint32_t n_sym, uint32_t ell)
        : K(CreateSL2ZFundamentalDomainTriangulation()),
          model(K),
          lssc(K, BuildRelations(K, n_sym, ell), n_sym + 1, ell),
          coh(&lssc) {
        coh.compute_cohomology();
    }

    static std::vector<LocalSystemSimplicialComplex::SimplexRelation>
    BuildRelations(const SimplicialComplex& K, uint32_t n_sym, uint32_t ell) {
        std::vector<LocalSystemSimplicialComplex::SimplexRelation> rels;
        for (const auto& datum : GenerateSL2ZRelationsWithGroupData(K, n_sym, ell)) {
            rels.push_back(datum.rel);
        }
        return rels;
    }
};

// ===========================================================================
// Arm 1 -- the ONE tri-state, its JSON spelling and its fixed reason list.
// ===========================================================================
void RunFinalTypeArm() {
    std::cout << "-- arm 1: the one tri-state `Final`, its migration and its reason list\n";

    // [X] Sec. 1.3.1 R3.2: "the tri-state is ONE type ... to which both units
    // then migrate".  SturmFinal IS Final since this package -- an alias, not
    // a second enum -- so a value produced by HeckeSturmBound is the value the
    // record layer consumes, with no conversion anywhere.
    static_assert(std::is_same<hecke_ext::SturmFinal, hecke_ext::Final>::value,
                  "SturmFinal must be an alias of Final");
    static_assert(std::is_same<hecke_ext::SturmFinality, hecke_ext::Finality>::value,
                  "SturmFinality must be an alias of Finality");
    Check(true, "SturmFinal / SturmFinality are aliases of Final / Finality");

    // The JSON strings are UNCHANGED by the migration.
    Check(std::string(hecke_ext::ToString(Final::True)) == "true", "ToString(True)");
    Check(std::string(hecke_ext::ToString(Final::False)) == "false", "ToString(False)");
    Check(std::string(hecke_ext::ToString(Final::Heuristic)) == "heuristic",
          "ToString(Heuristic)");
    Check(hecke_ext::ToJsonValue(Final::True) == "true", "ToJsonValue(True)");
    Check(hecke_ext::ToJsonValue(Final::False) == "false", "ToJsonValue(False)");
    Check(hecke_ext::ToJsonValue(Final::Heuristic) == "\"heuristic\"",
          "ToJsonValue(Heuristic) is a JSON string");

    // The fixed list of R3.2 plus E2a's record-level `not_applicable`.
    const char* good[] = {"", "not_local", "sturm_unavailable", "sturm_heuristic",
                          "closure_unreachable", "interrupted", "not_applicable"};
    for (const char* r : good) {
        Check(hecke_ext::IsFinalReason(r), std::string("IsFinalReason(\"") + r + "\")");
    }
    Check(hecke_ext::IsFinalReason(hecke_ext::CertificateFailedReason(17u)),
          "certificate_failed:17 is a legal reason");
    Check(hecke_ext::CertificateFailedReason(17u) == "certificate_failed:17",
          "CertificateFailedReason spelling");
    // WP-1 finding: the SHIPPED `generation_verdict` emits a certificate NAME
    // for route (c)'s negative answer, not a clause code.  Both forms are
    // legal; an invented name is not.
    for (const char* c : {"cyclic", "commutant", "sturm", "locality", "module", "identity"}) {
        Check(hecke_ext::IsFinalReason(hecke_ext::CertificateFailedReason(std::string(c))),
              std::string("certificate_failed:") + c + " is a legal reason");
    }
    Check(!hecke_ext::IsFinalReason("certificate_failed:made_up"),
          "an invented certificate name is not a legal reason");
    for (const char* k : {"sturm_bound", "locality", "commutant", "u_ell", "fill"}) {
        Check(hecke_ext::IsResumableKind(k), std::string("IsResumableKind(") + k + ")");
        Check(hecke_ext::IsFinalReason(hecke_ext::BudgetExhaustedReason(k)),
              std::string("budget_exhausted:") + k + " is a legal reason");
    }
    // ... and nothing else is.
    const char* bad[] = {"nope", "budget_exhausted:", "budget_exhausted:theta",
                         "certificate_failed:", "certificate_failed:x", "NOT_LOCAL",
                         "budget_exhausted", "certificate_failed"};
    for (const char* r : bad) {
        Check(!hecke_ext::IsFinalReason(r), std::string("IsFinalReason rejects \"") + r + "\"");
    }

    // Finality's structural invariant.
    Finality f;
    f.value = Final::True;
    Check(f.well_formed(), "a True verdict with no reason is well formed");
    f.reason = "not_local";
    Check(!f.well_formed(), "a True verdict WITH a reason is rejected");
    f.value = Final::False;
    Check(f.well_formed(), "a False verdict with a listed reason is well formed");
    f.reason.clear();
    Check(!f.well_formed(), "a False verdict with NO reason is rejected");
    f.reason = "made_up";
    Check(!f.well_formed(), "a False verdict with an unlisted reason is rejected");
}

// ===========================================================================
// Arm 2 -- the two new mathematical keys.
// ===========================================================================
void RunSettingsKeysArm() {
    std::cout << "-- arm 2: the `commutant_budget` and `final_mode` mathematical keys\n";

    const hecke::MathSettings& d = hecke::MathSettings::defaults();
    Check(d.is_default(), "the default MathSettings is default");
    CheckEq(d.fingerprint(), uint64_t(0), "the all-defaults fingerprint");
    CheckEq(d.commutant_budget.budget, hecke::kDefaultCommutantBudget,
            "commutant_budget's default");
    CheckEq(hecke::kDefaultCommutantBudget, uint64_t(20000000000ull),
            "kDefaultCommutantBudget = 2e10 ([X] Sec. 2.2 Stage 4(b))");
    CheckEq(CommutantRoute::kCommutantBudget, hecke::kDefaultCommutantBudget,
            "CommutantRoute::kCommutantBudget is an ALIAS, not a second literal");
    Check(d.final_mode == hecke::FinalMode::Closure, "final_mode's default is `closure`");

    // The registry order: APPENDED after `sturm_unavailable`, never inserted.
    const std::vector<std::string>& keys = hecke::MathSettings::keys();
    // (slot-5 WP-7 / E6) `field_polynomial` and `split_field_report` are
    // APPENDED after `final_mode`, both at their defaults -- so this list
    // grows at its TAIL and the all-defaults fingerprint checked above is
    // still 0.
    // (slot-7 WP-2) `presentation` and `predictions` APPENDED at the tail.
    // (slot-8 WP-3) `coefficient_ring` APPENDED at the tail.
    const std::vector<std::string> want = {"locality", "u_ell", "sturm", "sturm_unavailable",
                                           "commutant_budget", "final_mode", "field_polynomial",
                                           "split_field_report", "presentation", "predictions",
                                           "coefficient_ring"};
    Check(keys == want, "the registered key order");
    Check(hecke::MathSettings::known_key("commutant_budget"), "commutant_budget is registered");
    Check(hecke::MathSettings::known_key("final_mode"), "final_mode is registered");

    // Parsing, canonical text, round trip, fingerprint.
    {
        hecke::MathSettings s;
        Check(s.set("commutant_budget", "12345", hecke::SettingSource::Spec),
              "set(commutant_budget)");
        CheckEq(s.commutant_budget.budget, uint64_t(12345), "the parsed budget");
        Check(s.commutant_budget.to_text() == "12345", "commutant_budget round trip");
        Check(!s.is_default(), "a set commutant_budget is not the default");
        Check(s.fingerprint() != 0, "a non-default commutant_budget fingerprints non-zero");
        Check(s.canonical_text() == "commutant_budget = 12345\n", "the canonical text");
        const std::string j = s.to_json();
        Check(HasKey(j, "commutant_budget"), "to_json carries commutant_budget");
        Check(j.find("\"source\":\"spec\"") != std::string::npos, "to_json carries the source");
    }
    {
        hecke::MathSettings s;
        Check(s.set("final_mode", "sturm_heuristic", hecke::SettingSource::Console),
              "set(final_mode)");
        Check(s.final_mode == hecke::FinalMode::SturmHeuristic, "the parsed final_mode");
        Check(s.canonical_text() == "final_mode = sturm_heuristic\n",
              "final_mode's canonical text");
        Check(s.fingerprint() != 0, "a non-default final_mode fingerprints non-zero");
        Check(std::string(hecke::to_text(hecke::FinalMode::Closure)) == "closure",
              "to_text(Closure)");
    }
    // Setting a key back to its default returns the fingerprint to 0, so the
    // fingerprint is a function of the VALUES, not of the history.
    {
        hecke::MathSettings s;
        s.set("commutant_budget", "7", hecke::SettingSource::Spec);
        s.set("commutant_budget", std::to_string(hecke::kDefaultCommutantBudget),
              hecke::SettingSource::Spec);
        CheckEq(s.fingerprint(), uint64_t(0),
                "the fingerprint is a function of the values, not the history");
    }
    // Bad values are rejected with a named error, by the ENGINE and nowhere
    // else ([OD] R1: validated once).
    for (const char* bad : {"0", "", "-1", "1e9", "abc", "99999999999999999999"}) {
        hecke::MathSettings s;
        bool threw = false;
        try { s.set("commutant_budget", bad, hecke::SettingSource::Spec); }
        catch (const std::invalid_argument&) { threw = true; }
        Check(threw, std::string("commutant_budget rejects \"") + bad + "\"");
    }
    for (const char* bad : {"", "closure ", "heuristic", "sturm"}) {
        hecke::MathSettings s;
        bool threw = false;
        try { s.set("final_mode", bad, hecke::SettingSource::Spec); }
        catch (const std::invalid_argument&) { threw = true; }
        Check(threw, std::string("final_mode rejects \"") + bad + "\"");
    }
    // An unknown key is still `false`, not an exception (CampaignSpec's
    // convention).
    {
        hecke::MathSettings s;
        Check(!s.set("no_such_key", "x", hecke::SettingSource::Spec),
              "an unknown key returns false");
    }

    // The scoped override is thread_local and restores on scope exit.
    {
        Check(hecke::math_settings().is_default(), "the settings in force start at the defaults");
        {
            hecke::ScopedCommutantBudget guard(999);
            CheckEq(hecke::math_settings().commutant_budget.budget, uint64_t(999),
                    "ScopedCommutantBudget installs its value");
            Check(hecke::math_settings().fingerprint() != 0,
                  "the scoped non-default fingerprint is non-zero");
        }
        Check(hecke::math_settings().is_default(), "ScopedCommutantBudget restores");
    }
}

// ===========================================================================
// Arm 3 -- `is_final`'s truth table ([X] Sec. 1.3.1 R3.2).
// ===========================================================================
void RunIsFinalArm() {
    std::cout << "-- arm 3: is_final over final_mode x sturm_unavailable x "
                 "{proof range, heuristic range, none}\n";

    hecke::MathSettings def;                       // closure + nonfinal
    hecke::MathSettings heur_unavail;
    heur_unavail.set("sturm_unavailable", "heuristic", hecke::SettingSource::Spec);
    hecke::MathSettings heur_mode;
    heur_mode.set("final_mode", "sturm_heuristic", hecke::SettingSource::Spec);
    hecke::MathSettings both;
    both.set("sturm_unavailable", "heuristic", hecke::SettingSource::Spec);
    both.set("final_mode", "sturm_heuristic", hecke::SettingSource::Spec);

    auto base = []() {
        FinalInputs in;
        in.local = true;
        in.locality_complete = true;
        in.certificates_ok = true;
        return in;
    };

    auto expect = [](const Finality& f, Final v, const std::string& reason,
                     const std::string& what) {
        CheckEq(static_cast<uint32_t>(f.value), static_cast<uint32_t>(v), what + ": value");
        Check(f.reason == reason, what + ": reason = \"" + f.reason + "\", want \"" + reason + "\"");
        Check(f.well_formed(), what + ": well formed");
    };

    // --- the `true` routes ------------------------------------------------
    {
        FinalInputs in = base();
        in.cyclic_passed = true;
        expect(hecke_ext::is_final(in, def), Final::True, "", "(a) cyclic");
    }
    {
        FinalInputs in = base();
        in.commutant = CommutantOutcome::Passed;
        expect(hecke_ext::is_final(in, def), Final::True, "", "(b) commutant");
    }
    {
        FinalInputs in = base();
        in.sturm = SturmSweepOutcome::Passed;
        expect(hecke_ext::is_final(in, def), Final::True, "", "(c) sturm on its proof range");
    }

    // --- every reason of the fixed list -----------------------------------
    {
        FinalInputs in = base();
        in.not_applicable = true;
        expect(hecke_ext::is_final(in, def), Final::False, "not_applicable",
               "n = 0 / dim M_k = 0 ([E] Def. 2.8)");
        // ... and it wins over everything else, including a passing route.
        in.cyclic_passed = true;
        expect(hecke_ext::is_final(in, def), Final::False, "not_applicable",
               "not_applicable is not overridden by a certificate");
    }
    {
        FinalInputs in = base();
        in.interrupted = true;
        expect(hecke_ext::is_final(in, def), Final::False, "interrupted", "R3.3 item 5");
    }
    {
        FinalInputs in = base();
        in.local = false;
        in.locality_complete = true;
        expect(hecke_ext::is_final(in, def), Final::False, "not_local",
               "strict mode's proven non-locality");
        in.locality_complete = false;
        expect(hecke_ext::is_final(in, def), Final::False, "budget_exhausted:locality",
               "the budget survivor (R3.3 item 2: budget mode never PROVES non-local)");
    }
    {
        FinalInputs in = base();
        in.certificates_ok = false;
        in.first_failing_code = 23;
        expect(hecke_ext::is_final(in, def), Final::False, "certificate_failed:23",
               "a failing certificate clause");
    }
    {
        FinalInputs in = base();
        in.commutant = CommutantOutcome::OverBudget;
        expect(hecke_ext::is_final(in, def), Final::False, "budget_exhausted:commutant",
               "the budgeted commutant (R3.3 item 3)");
    }
    {
        FinalInputs in = base();
        in.commutant = CommutantOutcome::ExcludedBySocleBound;
        expect(hecke_ext::is_final(in, def), Final::False, "closure_unreachable",
               "the socle-bound PROOF that (b) fails");
        in.commutant = CommutantOutcome::Failed;
        expect(hecke_ext::is_final(in, def), Final::False, "closure_unreachable",
               "a computed dim C > dim A_S");
    }
    {
        FinalInputs in = base();
        in.sturm = SturmSweepOutcome::Failed;
        expect(hecke_ext::is_final(in, def), Final::False, "certificate_failed:sturm",
               "a swept T_q outside A_S");
    }
    {
        FinalInputs in = base();
        in.sturm = SturmSweepOutcome::Unavailable;
        expect(hecke_ext::is_final(in, def), Final::False, "sturm_unavailable",
               "no B' under the default mode");
    }

    // --- the two heuristic modes, the ONLY routes to "heuristic" ----------
    // (i) no B' (cap hit) ON the proof range: `sturm_unavailable = heuristic`.
    {
        FinalInputs in = base();
        in.sturm = SturmSweepOutcome::Heuristic;
        in.sturm_proof_range = true;
        in.sturm_has_B_prime = false;
        expect(hecke_ext::is_final(in, def), Final::False, "sturm_heuristic",
               "cap hit under the DEFAULT configuration -- never \"heuristic\"");
        expect(hecke_ext::is_final(in, heur_unavail), Final::Heuristic, "sturm_heuristic",
               "cap hit under sturm_unavailable = heuristic");
        expect(hecke_ext::is_final(in, heur_mode), Final::False, "sturm_heuristic",
               "cap hit is NOT final_mode's business");
        expect(hecke_ext::is_final(in, both), Final::Heuristic, "sturm_heuristic",
               "cap hit under both modes");
    }
    // (ii) a complete sweep OFF the proof range: `final_mode = sturm_heuristic`.
    {
        FinalInputs in = base();
        in.sturm = SturmSweepOutcome::Heuristic;
        in.sturm_proof_range = false;
        in.sturm_has_B_prime = true;
        expect(hecke_ext::is_final(in, def), Final::False, "sturm_heuristic",
               "the heuristic range under the DEFAULT configuration");
        expect(hecke_ext::is_final(in, heur_mode), Final::Heuristic, "sturm_heuristic",
               "the heuristic range under final_mode = sturm_heuristic");
        expect(hecke_ext::is_final(in, heur_unavail), Final::False, "sturm_heuristic",
               "the heuristic range is NOT sturm_unavailable's business");
        expect(hecke_ext::is_final(in, both), Final::Heuristic, "sturm_heuristic",
               "the heuristic range under both modes");
    }
    // The default configuration NEVER produces "heuristic", whatever else is
    // true ([X] R3.2, slot-1 confirmation 8).
    {
        uint32_t seen = 0;
        for (int na = 0; na < 2; ++na)
        for (int intr = 0; intr < 2; ++intr)
        for (int loc = 0; loc < 2; ++loc)
        for (int lcomp = 0; lcomp < 2; ++lcomp)
        for (int cok = 0; cok < 2; ++cok)
        for (int cyc = 0; cyc < 2; ++cyc)
        for (uint32_t b = 0; b <= 5; ++b)
        for (uint32_t c = 0; c <= 5; ++c)
        for (int pr = 0; pr < 2; ++pr)
        for (int hb = 0; hb < 2; ++hb) {
            FinalInputs in;
            in.not_applicable = (na != 0);
            in.interrupted = (intr != 0);
            in.local = (loc != 0);
            in.locality_complete = (lcomp != 0);
            in.certificates_ok = (cok != 0);
            in.first_failing_code = 5;
            in.cyclic_passed = (cyc != 0);
            in.commutant = static_cast<CommutantOutcome>(b);
            in.sturm = static_cast<SturmSweepOutcome>(c);
            in.sturm_proof_range = (pr != 0);
            in.sturm_has_B_prime = (hb != 0);
            const Finality f = hecke_ext::is_final(in, def);
            if (!f.well_formed()) ++seen;
            if (f.value == Final::Heuristic) ++seen;
        }
        CheckEq(seen, 0u,
                "over the whole 9216-point input grid the DEFAULT configuration produces "
                "neither a malformed verdict nor \"heuristic\"");
    }

    // (slot-12 FIN, audit G3) The finality certificates' branch ordering: a passed
    // (d0) / (d1) comes AFTER !local / !certificates_ok / interrupted / not_applicable
    // and BEFORE the heuristic modes (a proof is `true` in every mode).
    for (int route = 0; route < 2; ++route) {
        const std::string rn = route == 0 ? "(d0)" : "(d1)";
        auto passed = [&]() {
            FinalInputs in = base();
            in.cyclic_passed = false;
            in.commutant = CommutantOutcome::ExcludedBySocleBound;
            in.sturm = SturmSweepOutcome::Heuristic;
            in.sturm_proof_range = false;
            in.sturm_has_B_prime = true;
            (route == 0 ? in.dickson_passed : in.boundary_passed) = true;
            return in;
        };
        {
            FinalInputs in = passed();
            in.local = false;
            in.locality_complete = true;
            expect(hecke_ext::is_final(in, def), Final::False, hecke_ext::kReasonNotLocal,
                   rn + " passed + local = false -> not_local");
            in.locality_complete = false;
            expect(hecke_ext::is_final(in, def), Final::False,
                   hecke_ext::BudgetExhaustedReason(hecke_ext::kKindLocality),
                   rn + " passed + locality budget survivor -> budget_exhausted:locality");
        }
        {
            FinalInputs in = passed();
            in.certificates_ok = false;
            in.first_failing_code = 7;
            expect(hecke_ext::is_final(in, def), Final::False, "certificate_failed:7",
                   rn + " passed + certificates_ok = false -> certificate_failed:<code>");
        }
        {
            FinalInputs in = passed();
            in.interrupted = true;
            expect(hecke_ext::is_final(in, def), Final::False, hecke_ext::kReasonInterrupted,
                   rn + " passed + interrupted -> interrupted");
            in = passed();
            in.not_applicable = true;
            expect(hecke_ext::is_final(in, def), Final::False, hecke_ext::kReasonNotApplicable,
                   rn + " passed + not_applicable -> not_applicable");
        }
        for (const hecke::MathSettings* m : {&def, &heur_unavail, &heur_mode, &both}) {
            expect(hecke_ext::is_final(passed(), *m), Final::True, "",
                   rn + " passed, off the proof range, in every mode -> true");
        }
        {
            FinalInputs in = passed();
            (route == 0 ? in.dickson_passed : in.boundary_passed) = false;
            expect(hecke_ext::is_final(in, heur_mode), Final::Heuristic, hecke_ext::kReasonSturmHeuristic,
                   rn + " NOT passed, off range, sturm_heuristic mode -> \"heuristic\" (unchanged)");
            expect(hecke_ext::is_final(in, def), Final::False, hecke_ext::kReasonSturmHeuristic,
                   rn + " NOT passed, off range, closure -> false : sturm_heuristic (fall-through)");
        }
    }
}

// ===========================================================================
// Arm 4 -- the per-complex degree fields and the writer's refusals.
// ===========================================================================
void RunComplexFieldsArm() {
    std::cout << "-- arm 4: the per-complex degree fields ([C] sec:recordschema item 5)\n";

    HeckeExtRecord r;
    r.n_sym = 10;
    r.ell = 691;
    r.weight = 12;
    r.bulk.available = true;
    r.bulk.num_degrees = 3;       // the bulk model's own degree count

    const std::string j = hecke_ext::BuildHeckeExtReportJson(r);
    CheckEq(r.verify(), 0u, "a minimal record verifies");
    Check(j.find("\"bulk\":{\"status\":\"computed\",\"q_max\":2,\"truncated\":false,"
                 "\"q_max_source\":\"derived\",\"degrees\":[0,1,2]}") != std::string::npos,
          "the bulk complex's four derived fields");
    // A record that attached no cone keeps the placeholder of [C]
    // sec:recordschema item 6(c) -- "and nothing else", which arm 13 is the
    // negative control for.
    Check(j.find("\"boundary\":{\"status\":\"unavailable\"}") != std::string::npos,
          "an unattached boundary is the placeholder");
    Check(j.find("\"pair\":{\"status\":\"unavailable\"}") != std::string::npos,
          "an unattached pair is the placeholder");
    // The two cone flags belong INSIDE an available `pair`, so they are absent
    // here -- emitted, since v2, only where the complex they are about exists.
    Check(!HasKey(j, "phi_flat_identity"), "phi_flat_identity needs an available pair");
    Check(!HasKey(j, "residual_flat_zero"), "residual_flat_zero needs an available pair");

    // A truncated complex names its source; a `set` q_max on a complex that is
    // NOT truncated is refused, exactly as HeckeConeValidation pins for the
    // cone ([C] item 5: `set` only where truncated is true).
    {
        HeckeExtRecord t = r;
        t.bulk.truncated = true;
        t.bulk.q_max_configured = true;
        t.bulk.num_degrees = 2;
        const std::string tj = hecke_ext::BuildHeckeExtReportJson(t);
        Check(tj.find("\"q_max\":1,\"truncated\":true,\"q_max_source\":\"set\"") !=
                  std::string::npos,
              "a declared truncation records `set`");
        CheckEq(t.verify(), 0u, "a declared truncation verifies");
    }
    {
        HeckeExtRecord t = r;
        t.bulk.q_max_configured = true;   // but truncated == false
        CheckEq(t.verify(), 4u, "verify() rejects `set` on a complex that is not truncated");
        bool threw = false;
        try { hecke_ext::BuildHeckeExtReportJson(t); }
        catch (const std::exception&) { threw = true; }
        Check(threw, "the writer refuses `set` on a complex that is not truncated");
    }
    {
        HeckeExtRecord t = r;
        t.bulk.num_degrees = 0;           // available but nameless
        CheckEq(t.verify(), 3u, "verify() rejects an available complex with no degree count");
        bool threw = false;
        try { hecke_ext::BuildHeckeExtReportJson(t); }
        catch (const std::exception&) { threw = true; }
        Check(threw,
              "the writer REFUSES a record that cannot name its complex "
              "(the HeckeReportInput::num_degrees lesson)");
    }
    // Structural refusals of verify().
    {
        HeckeExtRecord t = r;
        t.weight = 11;
        CheckEq(t.verify(), 2u, "verify() rejects weight != n_sym + 2");
    }
    {
        HeckeExtRecord t = r;
        t.sign = "±";
        CheckEq(t.verify(), 5u, "verify() rejects an unknown sign");
    }
    {
        HeckeExtRecord t = r;
        t.iota_sign = "eisenstein_in_H1_minus";
        CheckEq(t.verify(), 15u, "verify() rejects a changed iota_sign constant");
    }
}

// ===========================================================================
// Arm 5 -- `not_applicable` at n = 0, and the key set of [E] Sec. 7.
// ===========================================================================
void RunKeySetArm() {
    std::cout << "-- arm 5: the [E] Sec. 7 key set, and `not_applicable` at n = 0\n";

    HeckeExtRecord r;
    r.n_sym = 0;
    r.ell = 691;
    r.weight = 2;
    r.bulk.available = true;
    r.bulk.num_degrees = 3;
    r.groups.push_back({"H0", true, 1});
    r.groups.push_back({"H1", true, 0});
    r.groups.push_back({"H1c", false, 0});
    r.operators_computed = {2, 3, 5, 7};
    r.certificates_ran = {"locality"};
    r.hashes.push_back({"A1_T2", 0x1234567890abcdefull});
    r.convention_present = true;
    r.tau_mod_ell_ok = true;
    r.eis_eigenvalue_ok = true;

    // [E] Def. 2.8: dim M_k = 0 at k = 2, so there is no claim to be final
    // about -- the RECORD-LEVEL state, while SturmOutcome stays three-valued
    // in code.
    IdealRecord id;
    id.id = 0;
    id.dim = 1;
    id.local = true;
    id.residue_degree = 1;
    id.residue_degree_certified = true;
    id.witness_poly = {1, 1};
    id.final_inputs.not_applicable = true;
    id.final_inputs.local = true;
    id.final = hecke_ext::is_final(id.final_inputs, r.settings);
    r.ideals.push_back(id);

    CheckEq(r.verify(), 0u, "the n = 0 record verifies");
    const std::string j = hecke_ext::BuildHeckeExtReportJson(r);
    Check(j.find("\"final\":false,\"final_reason\":\"not_applicable\"") != std::string::npos,
          "n = 0 is emitted as final = false with reason `not_applicable`");

    // Every top-level key of [E] Sec. 7 that WP-1 fills.
    const char* keys[] = {"schema", "commit", "model", "iota_sign", "n_sym", "ell", "weight",
                          "sign", "settings", "coefficient_ring", "resumable_items",
                          "convention_check", "complexes", "groups", "les",
                          "operators_computed", "ideals", "identities", "certificates_ran",
                          "hashes", "poly_coeff_order", "matrix_entry_order"};
    for (const char* k : keys) Check(HasKey(j, k), std::string("top-level key `") + k + "`");
    Check(j.find("\"schema\":\"hecke_ext/v5\"") != std::string::npos, "the schema string");
    Check(j.find("\"model\":\"sl2z_7vertex\"") != std::string::npos, "the model");
    Check(j.find("\"iota_sign\":\"eisenstein_in_H1_plus\"") != std::string::npos,
          "iota_sign is the constant of rmk:iotasign");
    Check(j.find("\"sign\":\"all\"") != std::string::npos,
          "sign = \"all\" (the per-sign values are slot 5's)");
    Check(j.find("\"coefficient_ring\":\"F_ell\"") != std::string::npos, "coefficient_ring");
    Check(j.find("\"fingerprint\":\"0000000000000000\"") != std::string::npos,
          "the settings fingerprint is 0 at the defaults");
    Check(j.find("\"dim_M_k\":") != std::string::npos, "the groups object carries dim_M_k");
    Check(j.find("\"H1c\":{\"status\":\"unavailable\"}") != std::string::npos,
          "an unavailable group is `unavailable`, never a guessed 0");
    Check(j.find("\"les\":{\"status\":\"unavailable\"}") != std::string::npos,
          "the LES summary is unavailable before slot 5");
    // Per-ideal keys.
    const char* ikeys[] = {"id", "dim", "residue_degree", "residue_degree_certified", "local",
                           "residue_witness", "witness_poly", "certificate",
                           "certificate_routes", "generators_certified", "boundary_reps", "tangent", "modules",
                           "canonical", "final", "final_reason"};
    for (const char* k : ikeys) Check(HasKey(j, k), std::string("per-ideal key `") + k + "`");
    Check(HexU64(0x1234567890abcdefull) == "1234567890abcdef", "the hash rendering");
    Check(j.find("\"A1_T2\":\"1234567890abcdef\"") != std::string::npos, "a recorded hash");

    // A-N4: `residue_degree` XOR `residue_degree_lower_bound`, never both.
    {
        HeckeExtRecord t = r;
        t.ideals[0].residue_degree_certified = false;
        const std::string tj = hecke_ext::BuildHeckeExtReportJson(t);
        Check(HasKey(tj, "residue_degree_lower_bound"),
              "an uncertified residue degree is a lower bound");
        Check(tj.find("\"residue_degree\":") == std::string::npos,
              "the two residue-degree keys are never both present");
    }
    // A-N4 again: the coefficient field is emitted only on a local block.
    {
        HeckeExtRecord t = r;
        t.ideals[0].local = false;
        t.ideals[0].coefficient_field.present = true;
        CheckEq(t.verify(), 13u, "verify() rejects coefficient_field on a non-local block");
    }
    // A record whose stored verdict is not is_final()'s own is refused.
    {
        HeckeExtRecord t = r;
        t.ideals[0].final.value = Final::True;
        t.ideals[0].final.reason.clear();
        CheckEq(t.verify(), 8u, "verify() rejects a hand-written final verdict");
        bool threw = false;
        try { hecke_ext::BuildHeckeExtReportJson(t); }
        catch (const std::exception&) { threw = true; }
        Check(threw, "the writer refuses a hand-written final verdict");
    }
}

// ===========================================================================
// Arm 6 -- the R3.1 item-6 sidecar.
// ===========================================================================
void RunSidecarArm() {
    std::cout << "-- arm 6: the R3.1 item 6 sidecar (path, FNV, reload)\n";

    // A state whose JSON is comfortably over the 64 KiB limit.
    std::string big = "{\"kind\":\"commutant\",\"filler\":\"";
    big.append(70000, 'x');
    big += "\"}";
    Check(big.size() > 65536u, "the forced state is oversize");

    HeckeExtRecord r;
    r.n_sym = 10;
    r.ell = 691;
    r.weight = 12;
    r.bulk.available = true;
    r.bulk.num_degrees = 3;

    IdealRecord id;
    id.id = 0;
    id.dim = 3;
    id.local = true;
    id.final_inputs.local = true;
    id.final_inputs.commutant = CommutantOutcome::OverBudget;
    id.final = hecke_ext::is_final(id.final_inputs, r.settings);
    ResumableRecord rr;
    rr.present = true;
    rr.kind = "commutant";
    rr.complete = false;
    rr.budget_json = "{\"modmul\":40}";
    rr.reached_json = "{\"rows_done\":2}";
    rr.budget_history_json = "[40]";
    rr.state_json = big;
    id.resumables.push_back(rr);
    r.ideals.push_back(id);

    CheckEq(r.resumable_items(), 1u, "resumable_items counts the incomplete item");
    CheckEq(r.verify(), 0u, "the record with a resumable item verifies");

    // (a) NO stem: the state is inlined and FLAGGED, never a dangling path
    //     (the slot-1 WP-3 rule).
    {
        const std::string j = hecke_ext::BuildHeckeExtReportJson(r);
        Check(HasKey(j, "state_oversize"), "no stem: `state_oversize` is set");
        Check(HasKey(j, "state_fnv64"), "no stem: the state's hash is recorded");
        Check(!HasKey(j, "state_ref"), "no stem: no path that names no file");
        Check(j.find(big) != std::string::npos, "no stem: the state is inlined anyway");
        Check(j.find("\"resumable_items\":1") != std::string::npos,
              "resumable_items is emitted");
    }

    // (b) WITH a stem (WriteHeckeExtReport): a sidecar beside the record,
    //     referenced by RELATIVE path and FNV-1a-64.
    {
        const std::string dir = TmpDir();
        const std::string base =
            "oc_extrec_" + std::to_string(static_cast<long>(::getpid())) + ".json";
        const std::string path = dir + "/" + base;
        hecke_ext::WriteHeckeExtReport(r, path);

        std::ifstream in(path.c_str(), std::ios::binary);
        Check(in.good(), "the record file was written");
        std::string text((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
        in.close();

        const std::string stem = base.substr(0, base.size() - 5);
        const std::string rel = stem + ".state.commutant.json";
        Check(text.find("\"state_ref\":{\"path\":\"" + rel + "\"") != std::string::npos,
              "the sidecar is referenced by its RELATIVE path (the basename)");
        Check(text.find(big) == std::string::npos, "the oversize state is NOT inlined too");
        Check(!HasKey(text, "state_oversize"), "with a stem there is no `state_oversize`");

        const uint64_t want_fnv = engine::FnvBytes(engine::kFnvOffset64, big);
        Check(text.find("\"fnv64\":\"" + HexU64(want_fnv) + "\"") != std::string::npos,
              "the sidecar's FNV-1a-64 is the one byte-wise FNV of Fnv.h");
        Check(text.find("\"bytes\":" + std::to_string(big.size())) != std::string::npos,
              "the sidecar's byte count");

        // The round trip: reload the sidecar and compare entrywise.
        std::ifstream sc((dir + "/" + rel).c_str(), std::ios::binary);
        Check(sc.good(), "the sidecar file exists beside the record");
        std::string reloaded((std::istreambuf_iterator<char>(sc)),
                             std::istreambuf_iterator<char>());
        sc.close();
        Check(reloaded == big, "the reloaded sidecar is the state, byte for byte");
        CheckEq(engine::FnvBytes(engine::kFnvOffset64, reloaded), want_fnv,
                "the reloaded sidecar hashes to the recorded fnv64");

        std::remove((dir + "/" + rel).c_str());
        std::remove(path.c_str());
    }

    // (c) A second state of the SAME kind gets the documented disambiguator.
    {
        HeckeExtRecord t = r;
        t.ideals.push_back(t.ideals[0]);
        t.ideals[1].id = 1;
        CheckEq(t.resumable_items(), 2u, "two incomplete items are counted");
        const std::string dir = TmpDir();
        const std::string base =
            "oc_extrec2_" + std::to_string(static_cast<long>(::getpid())) + ".json";
        const std::string path = dir + "/" + base;
        hecke_ext::WriteHeckeExtReport(t, path);
        std::ifstream in(path.c_str(), std::ios::binary);
        std::string text((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
        in.close();
        const std::string stem = base.substr(0, base.size() - 5);
        Check(text.find(stem + ".state.commutant.json") != std::string::npos,
              "the FIRST state of a kind keeps [X]'s literal name");
        Check(text.find(stem + ".state.commutant.1.json") != std::string::npos,
              "the SECOND gets the `.1` disambiguator");
        std::remove((dir + "/" + stem + ".state.commutant.json").c_str());
        std::remove((dir + "/" + stem + ".state.commutant.1.json").c_str());
        std::remove(path.c_str());
    }

    // A state kind outside R3.1's five is refused.
    {
        HeckeExtRecord t = r;
        t.ideals[0].resumables[0].kind = "theta";
        CheckEq(t.verify(), 9u, "verify() rejects an unlisted resumable kind");
        bool threw = false;
        try { hecke_ext::BuildHeckeExtReportJson(t); }
        catch (const std::exception&) { threw = true; }
        Check(threw, "the writer refuses an unlisted resumable kind");
    }
}

// ===========================================================================
// Arm 7 -- `commutant_budget` drives CommutantRoute, and its R3 resume.
// ===========================================================================
void RunCommutantBudgetArm() {
    std::cout << "-- arm 7: commutant_budget drives CommutantRoute, and the R3 resume\n";

    // The regular module of A = F_ell[eps]/eps^2 on V = F_ell^2, as the
    // single-operator family {(2, eps)}: k = 1, so the block is local for
    // free, and V is cyclic, so (a) passes and (b) is `NotNeeded` -- which is
    // why the arm forces the structured system (the documented validation
    // knob) to have a budget to exhaust at all.
    const uint32_t ell = 691;
    ExtMatrix eps(2, 2, ell);
    eps.data[1][0] = 1;
    std::vector<std::pair<uint64_t, ExtMatrix>> ops = {{2, eps}};
    hecke::HeckeModuleDecomposition d(ops, ell);
    d.compute();
    CheckEq(d.certify(), 0u, "the constructed decomposition certifies");
    Check(d.blocks().size() == 1, "the constructed family has one block");
    if (d.blocks().size() != 1) return;
    const LocalAlgebra L = hecke_ext::ComputeLocalAlgebra(d.blocks()[0], ops, ell);
    Check(L.ok, "the constructed block certifies local");
    if (!L.ok) return;
    CheckEq(L.dim_A, 2u, "A_S = F_ell[eps]/eps^2");

    // A default-budget route reads the SETTING ([OD] slot-3 confirmation 3).
    {
        CommutantRoute r(L, 0, true);
        CheckEq(r.budget(), hecke::kDefaultCommutantBudget,
                "budget 0 = the setting in force, which at the default is kCommutantBudget");
    }
    {
        hecke::ScopedCommutantBudget guard(123456789ull);
        CommutantRoute r(L, 0, true);
        CheckEq(r.budget(), uint64_t(123456789ull),
                "CommutantRoute READS the `commutant_budget` setting");
    }

    // The reference run, and the cost model re-derived from its own state.
    CommutantRoute full(L, CommutantRoute::kCommutantBudget, true);
    const CommutantOutcome fo = full.run();
    Check(fo == CommutantOutcome::Passed || fo == CommutantOutcome::Failed,
          "the reference run completed");
    const hecke_ext::CommutantState& fs = full.state();
    const uint64_t D = fs.D, k = fs.k, u = fs.u;
    const uint64_t cost_formation = 2ull * k * u * D * D * D;
    const uint64_t cost_row = u * u;
    CheckEq(fs.total_rows, k * D * D, "total_rows = k * D^2");
    Check(fs.total_rows >= 3, "the constructed system has at least three rows to cut");

    // A budget that stops after two rows, installed AS THE SETTING.
    const uint64_t small = cost_formation + 2ull * cost_row;
    const uint64_t big = CommutantRoute::kCommutantBudget;
    hecke_ext::CommutantState cut_state;
    hecke_ext::CommutantCertificate resumed_cert;
    {
        hecke::ScopedCommutantBudget guard(small);
        CommutantRoute r(L, 0, true);
        const CommutantOutcome out = r.run();
        CheckEq(static_cast<uint32_t>(out), static_cast<uint32_t>(CommutantOutcome::OverBudget),
                "the small budget cuts the system off");
        CheckEq(r.state().rows_done, uint64_t(2), "the cut is exactly where the budget says");
        cut_state = r.state();

        // R3.1 item 2: a budget below what the state has already reached is
        // REFUSED with the named error.
        bool threw = false;
        try { r.resume(small - 1); } catch (const std::exception&) { threw = true; }
        Check(threw, "a resume below the reached state is refused");

        // R3.1 item 4: a resume with a raised budget is a NEW RECORDED
        // SETTING -- the history grows and the last entry is the one in force.
        const CommutantOutcome done = r.resume(big);
        Check(done != CommutantOutcome::OverBudget, "the resume completed");
        resumed_cert = r.certificate();
        const std::vector<uint64_t> want_history = {small, big};
        Check(r.state().budget_history == want_history,
              "budget_history is [small, big] after run + resume");
        Check(r.state().equals(fs, /*ignore_budget_history=*/true),
              "run(small) + resume(big) == run(big), entrywise, modulo budget_history");
        Check(!r.state().equals(fs, /*ignore_budget_history=*/false),
              "the budget histories DO differ -- the one member that must");
    }
    CheckEq(resumed_cert.verify(), 0u, "the resumed certificate verifies");
    CheckEq(full.certificate().verify(), 0u, "the reference certificate verifies");

    // The raise is visible in the settings fingerprint, which is what makes it
    // "a new recorded setting" in the record's sense.
    {
        hecke::MathSettings a, b;
        a.set("commutant_budget", std::to_string(small), hecke::SettingSource::Console);
        b.set("commutant_budget", std::to_string(big), hecke::SettingSource::Console);
        Check(a.fingerprint() != b.fingerprint(),
              "two budgets have different fingerprints, so a resume is a new recorded setting");
        CheckEq(b.fingerprint(), uint64_t(0),
                "the raised budget IS the default here, so its fingerprint is 0");
    }
    // The cut state is a prefix of the full one.
    Check(cut_state.rows_done < fs.rows_done, "the cut state is a strict prefix");
}

// ===========================================================================
// Arm 8 -- the (10, 691) bulk record from the shipped pieces.
// ===========================================================================
void RunRealCellArm() {
    std::cout << "-- arm 8: the (10, 691) bulk record, built from the shipped pieces\n";

    const uint32_t n = 10, ell = 691;

    // `u_ell = off`, exactly as the golden harness runs ([X] Sec. 1.6): no
    // U_691 is built, so the arm stays inside the `fast` tier's budget.  The
    // setting is RECORDED in the record below, which is the point of
    // [X] Sec. 1.5 rule 2.
    hecke::UEllSetting off;
    off.mode = hecke::UEllMode::Off;
    const hecke::ScopedUEll u_guard(off);

    const hecke_ext::StageOneOperators s1 = hecke_ext::ComputeStageOneOperators(n, ell);
    const std::vector<uint64_t> want_primes = {2, 3, 5, 7};
    Check(s1.primes == want_primes, "S_0 = {2,3,5,7} under u_ell = off");
    Check(!s1.includes_u_ell(), "U_691 is not in S_0 under u_ell = off");

    Stack stack(n, ell);
    hecke::HeckeOperatorFamily family(&stack.coh, stack.K, stack.model, ell, n);
    const std::vector<uint32_t>& dims = stack.coh.get_H_eq_dimensions();
    const uint32_t degree = 1;
    Check(dims.size() > degree, "(10, 691): no H^1");
    if (dims.size() <= degree) return;
    CheckEq(dims[degree], 3u, "(10, 691): dim H^1");

    std::vector<std::pair<uint64_t, ExtMatrix>> ops;
    for (uint64_t q : s1.primes) ops.push_back({q, family.T_p(q)[degree]});

    hecke::HeckeModuleDecomposition decomp(ops, ell);
    decomp.compute();
    CheckEq(decomp.certify(), 0u, "(10, 691): the decomposition certifies");
    CheckEq(static_cast<uint32_t>(decomp.blocks().size()), 1u, "(10, 691): one block");
    if (decomp.blocks().size() != 1) return;
    const hecke::PrimaryBlock& block = decomp.blocks()[0];
    Check(block.local_certified, "(10, 691): the block is certified local");

    const LocalAlgebra L = hecke_ext::ComputeLocalAlgebra(block, ops, ell);
    Check(L.ok, "(10, 691): ComputeLocalAlgebra");
    if (!L.ok) return;
    CheckEq(L.D, 3u, "(10, 691): D");
    CheckEq(L.f, 1u, "(10, 691): f");
    CheckEq(L.dim_A, 2u, "(10, 691): A_S = F_691[eps]/eps^2");
    CheckEq(L.nu, 2u, "(10, 691): the nilpotency index");
    Check(L.gorenstein, "(10, 691): A_S is Gorenstein");
    Check(!L.cyclicity.cyclic, "(10, 691): (a) fails -- V is not cyclic");

    // --- Stage 5 --------------------------------------------------------
    const ModuleStructure ms = hecke_ext::ComputeModuleStructure(L);
    Check(ms.ok, "(10, 691): Stage 5 ran");
    CheckEq(ms.verify(), 0u, "(10, 691): ModuleStructure::verify()");
    // [E] Sec. 7's worked example at (10, 691), term for term.
    CheckVec(ms.loewy, {2, 1}, "(10, 691): the Loewy layers");
    CheckVec(ms.socle, {2, 1}, "(10, 691): the socle layers");
    CheckEq(ms.min_generators, 2u, "(10, 691): min_generators");
    Check(!ms.cyclic, "(10, 691): V is not cyclic");
    Check(!ms.free_rank_one, "(10, 691): V is not free of rank one");
    CheckEq(static_cast<uint32_t>(ms.structure_maps.size()), 1u,
            "(10, 691): one consecutive layer pair");
    if (ms.structure_maps.size() == 1) {
        const hecke_ext::StructureMap& sm = ms.structure_maps[0];
        CheckEq(sm.top_layer, 0u, "(10, 691): the piece is [0, 1]");
        CheckEq(sm.bottom_layer, 1u, "(10, 691): the piece is [0, 1]");
        CheckEq(sm.top_dim, 2u, "(10, 691): the top layer's K-dimension");
        CheckEq(sm.bottom_dim, 1u, "(10, 691): the bottom layer's K-dimension");
        CheckEq(sm.rank_over_f, 1u, "(10, 691): the structure map's K-rank");
        CheckEq(static_cast<uint32_t>(sm.matrices.size()), 4u,
                "(10, 691): one matrix per generator");
        for (const ExtMatrix& M : sm.matrices) {
            CheckEq(M.rows, 1u, "(10, 691): rows = the BOTTOM-layer basis");
            CheckEq(M.cols, 2u, "(10, 691): columns = the TOP-layer basis");
        }
    }
    // A perturbed structure-map matrix is refused by verify().
    {
        ModuleStructure t = ms;
        if (!t.structure_maps.empty() && !t.structure_maps[0].matrices.empty()) {
            ExtMatrix& M = t.structure_maps[0].matrices[0];
            M.data[0][0] = (M.data[0][0] + 1) % ell;
            CheckEq(t.verify(), 13u, "a perturbed structure-map matrix fails clause 13");
        }
        ModuleStructure t2 = ms;
        if (!t2.loewy.empty()) {
            t2.loewy[0] += 1;
            CheckEq(t2.verify(), 5u, "a perturbed Loewy layer fails clause 5");
        }
        ModuleStructure t3 = ms;
        if (!t3.socle.empty()) {
            t3.socle.pop_back();
            CheckEq(t3.verify(), 9u, "a truncated socle series fails clause 9");
        }
        ModuleStructure t4 = ms;
        if (!t4.jordan.empty() && !t4.jordan[0].empty()) {
            t4.jordan[0][0] += 1;
            CheckEq(t4.verify(), 10u, "a perturbed Jordan type fails clause 10");
        }
    }

    // --- the generation routes ------------------------------------------
    CommutantRoute cr(L, CommutantRoute::kCommutantBudget, false);
    const CommutantOutcome b = cr.run();
    CheckEq(static_cast<uint32_t>(b),
            static_cast<uint32_t>(CommutantOutcome::ExcludedBySocleBound),
            "(10, 691): (b) is excluded by the socle bound");

    hecke_ext::BlockRestrictionSource src(
        block.basis_columns,
        hecke_ext::FamilyOperatorSupplier(family, degree, ell, s1.includes_u_ell()));
    hecke_ext::SturmSweepOptions so = hecke_ext::SturmSweepOptions::from_settings(n + 2);
    hecke_ext::SturmSweep sw(L, n, src, so);
    const SturmSweepOutcome c = sw.run();
    CheckEq(static_cast<uint32_t>(c), static_cast<uint32_t>(SturmSweepOutcome::Passed),
            "(10, 691): (c) passes on its proof range");
    const hecke_ext::SturmCertificate cert = sw.certificate();
    CheckEq(cert.verify(), 0u, "(10, 691): the Sturm certificate verifies");
    CheckEq(cert.B_prime, 2u, "(10, 691): B' = 2");
    const std::vector<uint64_t> want_gen = {2};
    Check(cert.generators_certified == want_gen, "(10, 691): generators_certified = [2]");

    // --- the record -------------------------------------------------------
    HeckeExtRecord r;
    r.n_sym = n;
    r.ell = ell;
    r.weight = n + 2;
    r.settings = hecke::math_settings();
    r.bulk.available = true;
    r.bulk.num_degrees = static_cast<uint32_t>(dims.size());
    r.groups.push_back({"H0", true, dims.size() > 0 ? dims[0] : 0u});
    r.groups.push_back({"H1", true, dims[1]});
    r.dim_M_k = cert.dim_M_k;
    r.operators_computed = s1.primes;
    r.certificates_ran = {"locality", "cyclic", "commutant", "sturm"};

    IdealRecord id;
    id.id = 0;
    id.dim = L.D;
    id.residue_degree = block.residue_degree;
    id.residue_degree_certified = block.residue_degree_certified;
    id.local = block.local_certified;
    id.residue_witness = block.residue_witness;
    id.witness_poly = L.witness_poly.coeffs;
    id.coefficient_field.present = true;
    id.coefficient_field.w_hat_exponent = L.N;
    for (size_t i = 0; i < L.primes.size(); ++i) {
        id.coefficient_field.primes.push_back(L.primes[i]);
        id.coefficient_field.P.push_back(L.character[i].coeffs);
    }
    id.algebra_present = true;
    id.algebra_dim = L.dim_A;
    id.embdim = L.embdim;
    id.nilpotency = L.nu;
    id.gorenstein = L.gorenstein;
    id.hilbert = L.hilbert;
    id.certificate = "sturm(B')";
    id.certificate_routes.a = "not_cyclic";
    id.certificate_routes.b = hecke_ext::ToString(b);
    id.certificate_routes.c = "found";
    id.generators_certified = cert.generators_certified;
    id.sturm.present = true;
    id.sturm.has_B_prime = cert.has_B_prime;
    id.sturm.B_prime = cert.B_prime;
    id.sturm.cap = cert.cap;
    id.sturm.closure_dim = cert.closure_dim;
    id.sturm.dim_M_k = cert.dim_M_k;
    id.sturm.mode = hecke_ext::ToString(cert.mode);
    id.sturm.outcome = hecke_ext::ToString(cert.outcome);
    id.sturm.sturm_primes = cert.sturm_primes;
    id.sturm.proof_range = cert.proof_range;
    // v2: the U_ell verdict is the CELL's, not the ideal's.
    r.u_ell.present = true;
    r.u_ell.included = s1.u_ell.included;
    r.u_ell.reason = hecke_ext::to_string(s1.u_ell.reason);
    r.u_ell.predicted_mb = s1.u_ell.predicted_mb;
    r.u_ell.budget_mb = s1.u_ell.budget_mb;
    r.u_ell.model_version = s1.u_ell.model_version;
    for (size_t i = 0; i < L.primes.size(); ++i) {
        id.tangent_primes.push_back(L.primes[i]);
        id.tangent_active.push_back(L.active[i]);
        id.operator_primes.push_back(L.primes[i]);
        id.t_p.push_back(L.t[i]);
    }
    id.modules.push_back({"H1", ms});
    // (slot-7 WP-3, item 6) under the default split_field_report = canonical
    // a local ideal carries its canonical split (verify() code 28).
    {
        const hecke_ext::CanonicalSplit sp = hecke_ext::SplitInCanonicalField(
                ell, L.witness_poly, L.primes, L.character);
        id.split_present = true;
        id.split_ok = sp.ok;
        id.split_r = sp.r;
        id.split_fragment = sp.to_json();
    }
    id.final_inputs.local = true;
    id.final_inputs.certificates_ok = true;
    id.final_inputs.cyclic_passed = L.cyclicity.cyclic;
    id.final_inputs.commutant = b;
    id.final_inputs.sturm = c;
    id.final_inputs.sturm_proof_range = cert.proof_range;
    id.final_inputs.sturm_has_B_prime = cert.has_B_prime;
    id.final = hecke_ext::is_final(id.final_inputs, r.settings);
    r.ideals.push_back(id);

    CheckEq(static_cast<uint32_t>(r.ideals[0].final.value), static_cast<uint32_t>(Final::True),
            "(10, 691): final = true");
    Check(r.ideals[0].final.reason.empty(), "(10, 691): a `true` explains nothing");
    CheckEq(r.verify(), 0u, "(10, 691): the record verifies");
    CheckEq(r.resumable_items(), 0u, "(10, 691): nothing is incomplete");

    const std::string j = hecke_ext::BuildHeckeExtReportJson(r);
    Check(j.find("\"final\":true,\"final_reason\":null") != std::string::npos,
          "(10, 691): the emitted verdict");
    Check(j.find("\"certificate\":\"sturm(B')\"") != std::string::npos,
          "(10, 691): the emitted certificate");
    Check(j.find("\"B_prime\":2") != std::string::npos, "(10, 691): the emitted B'");
    Check(j.find("\"loewy\":[2,1]") != std::string::npos, "(10, 691): the emitted Loewy series");
    Check(j.find("\"piece\":[0,1]") != std::string::npos, "(10, 691): the emitted piece");
    // The settings block records the non-default `u_ell = off` this arm runs
    // under, with its source and a NON-ZERO fingerprint -- [X] Sec. 1.5 rule 2.
    Check(j.find("\"u_ell\":{\"value\":\"off\"") != std::string::npos,
          "(10, 691): the settings block records u_ell = off");
    Check(j.find("\"fingerprint\":\"0000000000000000\"") == std::string::npos,
          "(10, 691): a non-default setting fingerprints non-zero");

    // Determinism: the same record twice is the same bytes, and the three
    // collectors agree.
    Check(hecke_ext::BuildHeckeExtReportJson(r) == j, "the writer is deterministic");
    const uint64_t h_ideals = CollectExtIdeals(r);
    const uint64_t h_certify = CollectExtCertify(r);
    const uint64_t h_json = CollectExtJsonCanonical(r);
    CheckEq(CollectExtIdeals(r), h_ideals, "ext_ideals^1 is stable");
    CheckEq(CollectExtCertify(r), h_certify, "ext_certify^1 is stable");
    CheckEq(CollectExtJsonCanonical(r), h_json, "ext_json_canonical^1 is stable");
    std::cout << "EXT_RECORD cell=(10,691) ideals=" << r.ideals.size()
              << " ext_ideals^1=0x" << HexU64(h_ideals)
              << " ext_certify^1=0x" << HexU64(h_certify)
              << " ext_json_canonical^1=0x" << HexU64(h_json) << std::endl;
}

// ===========================================================================
// Arm 9 -- the campaign job id and shape signature.
// ===========================================================================
void RunJobIdArm() {
    std::cout << "-- arm 9: Job::id()'s `_s<fingerprint>` suffix and the shape signature\n";

    campaign::Job base;
    base.n_sym = 16;
    base.ell = 5;
    base.p = 2;
    base.r = 1;
    base.degrees = campaign::ModelDefaultDegrees("sl2z_7vertex");

    // The DEFAULT id is byte-unchanged ([X] Sec. 1.5 rule 3, exit item 4).
    Check(base.id() == "n16_ell5_p2_r1", "the default job id is byte-unchanged: " + base.id());
    Check(base.settings_fingerprint == 0, "a job's fingerprint defaults to 0");
    Check(base.shape().to_string().find(" s=") == std::string::npos,
          "the default shape signature carries no ` s=`");

    // A non-default mathematical setting appends the FULL 64-bit fingerprint,
    // after any `_q` tail.
    hecke::MathSettings m;
    m.set("commutant_budget", "12345", hecke::SettingSource::Spec);
    const uint64_t fp = m.fingerprint();
    Check(fp != 0, "the non-default fingerprint is non-zero");

    campaign::Job with = base;
    with.settings_fingerprint = fp;
    Check(with.id() == "n16_ell5_p2_r1_s" + HexU64(fp),
          "the non-default job id carries `_s<16 hex>`: " + with.id());
    Check(with.id() != base.id(), "the two ids differ");
    Check(with.shape().to_string() != base.shape().to_string(),
          "the two shape signatures differ");
    Check(with.shape().to_string().find(" s=" + HexU64(fp)) != std::string::npos,
          "the shape signature carries ` s=<fingerprint>`");
    Check(!(with.shape() == base.shape()), "the two shapes are not equal");
    Check(("jobs/" + with.id() + ".json") != ("jobs/" + base.id() + ".json"),
          "the two resume paths differ");

    // The suffix comes AFTER the `_q` tail, so the injectivity argument of
    // CampaignSpec.h extends verbatim.
    campaign::Job q = base;
    q.degrees = {1};
    Check(q.id() == "n16_ell5_p2_r1_q1", "the `_q` tail is unchanged: " + q.id());
    campaign::Job qs = q;
    qs.settings_fingerprint = fp;
    Check(qs.id() == "n16_ell5_p2_r1_q1_s" + HexU64(fp),
          "`_s` follows `_q`: " + qs.id());

    // Two different settings give two different ids (16 hex digits, never a
    // truncation, so the map fingerprint -> tail is injective).
    hecke::MathSettings m2;
    m2.set("final_mode", "sturm_heuristic", hecke::SettingSource::Spec);
    campaign::Job other = base;
    other.settings_fingerprint = m2.fingerprint();
    Check(other.id() != with.id(), "two different settings give two different ids");

    // A spec that sets a mathematical key parses it, and one that does not
    // plans byte-identical jobs.
    {
        const std::string text =
            "name = t\nn_sym = 16\nell = 5\np = 2\nr = 1\n";
        std::istringstream in(text);
        const campaign::CampaignSpec spec = campaign::ParseCampaignSpec(in, "<string>");
        CheckEq(spec.math.fingerprint(), uint64_t(0), "a plain spec is at the defaults");
        const std::vector<campaign::JobGroup> g = campaign::plan_campaign(spec);
        Check(g.size() == 1 && g[0].jobs.size() == 1, "the plain spec plans one job");
        if (g.size() == 1 && g[0].jobs.size() == 1) {
            Check(g[0].jobs[0].id() == "n16_ell5_p2_r1",
                  "the planned default id: " + g[0].jobs[0].id());
        }
    }
    {
        const std::string text =
            "name = t\nn_sym = 16\nell = 5\np = 2\nr = 1\ncommutant_budget = 12345\n";
        std::istringstream in(text);
        const campaign::CampaignSpec spec = campaign::ParseCampaignSpec(in, "<string>");
        CheckEq(spec.math.commutant_budget.budget, uint64_t(12345),
                "the spec parsed the mathematical key");
        CheckEq(spec.math.fingerprint(), fp, "the spec's fingerprint");
        const std::vector<campaign::JobGroup> g = campaign::plan_campaign(spec);
        Check(g.size() == 1 && g[0].jobs.size() == 1, "the spec plans one job");
        if (g.size() == 1 && g[0].jobs.size() == 1) {
            Check(g[0].jobs[0].id() == "n16_ell5_p2_r1_s" + HexU64(fp),
                  "the planned non-default id: " + g[0].jobs[0].id());
        }
    }
    // A bad value for a registered mathematical key is an ordinary spec error
    // with the ENGINE's own message ([OD] R1: validated once, never twice).
    {
        const std::string text =
            "name = t\nn_sym = 16\nell = 5\np = 2\nr = 1\ncommutant_budget = 0\n";
        std::istringstream in(text);
        bool threw = false;
        std::string msg;
        try { campaign::ParseCampaignSpec(in, "<string>"); }
        catch (const std::runtime_error& e) { threw = true; msg = e.what(); }
        Check(threw, "a bad mathematical value is a spec error");
        Check(msg.find("commutant_budget") != std::string::npos,
              "the spec error names the key: " + msg);
    }
    // An unknown key is still an unknown key.
    {
        const std::string text =
            "name = t\nn_sym = 16\nell = 5\np = 2\nr = 1\nno_such = 1\n";
        std::istringstream in(text);
        bool threw = false;
        try { campaign::ParseCampaignSpec(in, "<string>"); }
        catch (const std::runtime_error&) { threw = true; }
        Check(threw, "an unknown key is still rejected");
    }
}

// ===========================================================================
// Arm 10 -- the settings under concurrency.
//
// Two halves, both of which the CheckThread (ThreadSanitizer) run of this
// package inspects:
//
//   (a) THREAD-LOCAL ISOLATION.  Two std::threads install two different
//       `commutant_budget` values through ScopedCommutantBudget and each runs
//       its own CommutantRoute; neither sees the other's budget, and both get
//       the same dim_C, because the budget decides only WHERE the fixed row
//       sequence stops.
//
//   (b) PROPAGATION ONTO run_concurrently() TASKS, through the campaign.
//       `engine::tape::run_concurrently` cannot be called from a HeckeExt/ TU
//       at all -- check_heckeext_includes.sh rule 2 forbids `Engine/` here,
//       which is why slot-3 WP-1 put `u_ell`'s propagation arm in a `Hecke/`
//       validation.  The campaign is the in-scope route that exercises the
//       same propagator: RunCampaignSpec runs `jobs` job threads, each of
//       which installs the spec's MathSettings with ScopedMathSettings and
//       then fans its own pipeline out with run_concurrently(), so a task
//       thread that did NOT inherit the settings would be visible here.
// ===========================================================================
void RunConcurrencyArm() {
    std::cout << "-- arm 10: the settings under concurrency (thread-local isolation, "
                 "and a jobs = 4 campaign with a non-default key)\n";

    // --- (a) two budgets on two threads ----------------------------------
    const uint32_t ell = 691;
    ExtMatrix eps(2, 2, ell);
    eps.data[1][0] = 1;
    std::vector<std::pair<uint64_t, ExtMatrix>> ops = {{2, eps}};
    hecke::HeckeModuleDecomposition d(ops, ell);
    d.compute();
    if (d.blocks().size() != 1) { Check(false, "arm 10: the constructed family has one block"); return; }
    const LocalAlgebra L = hecke_ext::ComputeLocalAlgebra(d.blocks()[0], ops, ell);
    if (!L.ok) { Check(false, "arm 10: the constructed block certifies local"); return; }

    uint64_t seen_a = 0, seen_b = 0;
    uint32_t dim_a = 0, dim_b = 0;
    bool leaked_a = false, leaked_b = false;
    {
        std::thread ta([&] {
            const hecke::ScopedCommutantBudget guard(1000000007ull);
            CommutantRoute r(L, 0, true);
            seen_a = r.budget();
            r.run();
            dim_a = r.dim_C();
            leaked_a = (hecke::math_settings().commutant_budget.budget != 1000000007ull);
        });
        std::thread tb([&] {
            const hecke::ScopedCommutantBudget guard(2000000011ull);
            CommutantRoute r(L, 0, true);
            seen_b = r.budget();
            r.run();
            dim_b = r.dim_C();
            leaked_b = (hecke::math_settings().commutant_budget.budget != 2000000011ull);
        });
        ta.join();
        tb.join();
    }
    CheckEq(seen_a, uint64_t(1000000007ull), "thread A read its own commutant_budget");
    CheckEq(seen_b, uint64_t(2000000011ull), "thread B read its own commutant_budget");
    Check(!leaked_a && !leaked_b, "the two threads' budgets did not leak into each other");
    CheckEq(dim_a, dim_b, "the budget decides WHERE the row loop stops, never dim C");
    Check(hecke::math_settings().is_default(),
          "the launching thread's settings are still the defaults");

    // --- (b) a jobs = 4 campaign with one non-default mathematical key ----
    namespace fs = std::filesystem;
    std::error_code ec;
    const fs::path base = fs::temp_directory_path(ec);
    const fs::path dir = (ec ? fs::path(".") : base) /
                         ("oc_extrec_campaign_" + std::to_string(static_cast<long>(::getpid())));
    fs::remove_all(dir, ec);
    fs::create_directories(dir, ec);

    hecke::MathSettings m;
    m.set("commutant_budget", "12345", hecke::SettingSource::Spec);
    const std::string tag = HexU64(m.fingerprint());

    const std::string text =
        "name = ext_record_concurrency\n"
        "n_sym = 2,4,6,8\n"
        "ell = 5\n"
        "p = 2\n"
        "r = 1\n"
        "jobs = 4\n"
        "threads = 1\n"
        "certify = none\n"
        "commutant_budget = 12345\n";
    std::istringstream is(text);
    campaign::CampaignSpec spec = campaign::ParseCampaignSpec(is, "<ext-record>");
    spec.output_dir = dir.string();
    CheckEq(spec.math.fingerprint(), m.fingerprint(), "the spec's fingerprint");

    const campaign::CampaignResult res = campaign::RunCampaignSpec(spec, /*verbose=*/false);
    CheckEq(static_cast<uint32_t>(res.jobs.size()), 4u, "the campaign ran four jobs");
    uint32_t ok_jobs = 0;
    for (const campaign::JobResult& jr : res.jobs) {
        if (jr.ok) ++ok_jobs;
        Check(jr.job.settings_fingerprint == m.fingerprint(),
              "a job did not carry the campaign's settings fingerprint");
        Check(jr.job.id().find("_s" + tag) != std::string::npos,
              "a job id lost its `_s<fingerprint>` suffix: " + jr.job.id());
        std::ifstream in(jr.output_path.c_str(), std::ios::binary);
        std::string j((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
        in.close();
        Check(j.find("\"commutant_budget\":{\"value\":\"12345\",\"source\":\"spec\"}") !=
                  std::string::npos,
              "a per-job JSON lost its `settings` block");
        Check(j.find(" s=" + tag) != std::string::npos,
              "a per-job shape_signature lost its ` s=<fingerprint>`");
    }
    CheckEq(ok_jobs, 4u, "all four jobs completed");

    // The same grid at the DEFAULTS writes the pre-E2a file names.
    const std::string plain_text =
        "name = ext_record_concurrency_default\n"
        "n_sym = 2,4,6,8\n"
        "ell = 5\n"
        "p = 2\n"
        "r = 1\n"
        "jobs = 4\n"
        "threads = 1\n"
        "certify = none\n";
    std::istringstream pis(plain_text);
    campaign::CampaignSpec plain = campaign::ParseCampaignSpec(pis, "<ext-record>");
    plain.output_dir = (dir / "plain").string();
    fs::create_directories(plain.output_dir, ec);
    const campaign::CampaignResult pres = campaign::RunCampaignSpec(plain, /*verbose=*/false);
    CheckEq(static_cast<uint32_t>(pres.jobs.size()), 4u, "the default campaign ran four jobs");
    for (const campaign::JobResult& jr : pres.jobs) {
        Check(jr.job.id().find("_s") == std::string::npos,
              "a default job id grew a settings suffix: " + jr.job.id());
        Check(jr.job.shape().to_string().find(" s=") == std::string::npos,
              "a default shape signature grew ` s=`");
    }
    Check(pres.jobs[0].job.id() == "n2_ell5_p2_r1",
          "the default job id is byte-unchanged: " + pres.jobs[0].job.id());

    fs::remove_all(dir, ec);
}

// ===========================================================================
// The slot-5 WP-1 golden COLLECTORS for the `hecke_ext/v2` additions.
//
// [X] Sec. 1.6's tier-A list, for the objects v2 adds.  DEFINED HERE AND
// CALLED HERE ONLY: slot-5 WP-1 emits no golden entry (the slot's one emission
// is WP-3's, and it must come AFTER this schema bump -- slot-5 plan Sec. 4.4
// rule 1), so none of these may be reachable from Golden/GoldenHarness.cpp's
// RunCell32 / RunCell64.  They are file-local for exactly that reason, and
// WP-3 is the package that moves them into the harness.
//
// Each returns an FNV-1a-64 over the record's OWN fields, so a collector is a
// function of the emitted record and not of the objects behind it -- which is
// what makes it checkable by a third party holding the JSON alone.
// ===========================================================================

// From E3 ([X] Sec. 1.6): charpoly_iota^q, rank_iota^q, dim_Hplus^q,
// dim_Hminus^q, eisenstein_line_defined.
uint64_t CollectIota(const HeckeExtRecord& r) {
    uint64_t h = engine::kFnvOffset64;
    for (const hecke_ext::IotaDegreeRecord& ir : r.iota) {
        h = engine::FnvU64(h, ir.degree);
        h = engine::FnvBytes(h, ir.status);
        h = engine::FnvU64(h, ir.dim_H);
        h = engine::FnvU64(h, ir.dim_plus);
        h = engine::FnvU64(h, ir.dim_minus);
        h = engine::FnvU64(h, ir.rank_iota);
        for (uint32_t c : ir.charpoly_iota) h = engine::FnvU64(h, c);
        h = engine::FnvBytes(h, ir.eisenstein_status);
        h = engine::FnvU64(h, ir.eisenstein_dim_socle_over_F);
    }
    return h;
}

// From E4 ([X] Sec. 1.6): dim_H0_bd, dim_H1_bd, rank_r^q -- as the RECORD
// carries them (the groups and `les.ranks`).
uint64_t CollectBoundaryDims(const HeckeExtRecord& r) {
    uint64_t h = engine::kFnvOffset64;
    for (const hecke_ext::GroupRecord& g : r.groups) {
        if (g.name.size() < 3) continue;
        if (g.name.compare(g.name.size() - 3, 3, "_bd") != 0) continue;
        h = engine::FnvBytes(h, g.name);
        h = engine::FnvU64(h, g.available ? g.dim : 0xffffffffu);
    }
    for (const std::pair<std::string, uint32_t>& rk : r.les_ranks) {
        h = engine::FnvBytes(h, rk.first);
        h = engine::FnvU64(h, rk.second);
    }
    h = engine::FnvU64(h, r.has_dim_H1_shriek ? r.dim_H1_shriek : 0xffffffffu);
    return h;
}

// From E5 ([X] Sec. 1.6): dim_H1c, dim_H2c, les_exact, and the recorded cone
// flags -- again as the RECORD carries them.
uint64_t CollectConeDims(const HeckeExtRecord& r) {
    uint64_t h = engine::kFnvOffset64;
    for (const hecke_ext::GroupRecord& g : r.groups) {
        if (g.name.empty() || g.name[g.name.size() - 1] != 'c') continue;
        h = engine::FnvBytes(h, g.name);
        h = engine::FnvU64(h, g.available ? g.dim : 0xffffffffu);
    }
    h = engine::FnvU64(h, r.les_available ? 1u : 0u);
    h = engine::FnvU64(h, r.les_exact ? 1u : 0u);
    h = engine::FnvU64(h, r.pair.has_cone_flags ? 1u : 0u);
    h = engine::FnvU64(h, r.pair.phi_flat_identity ? 1u : 0u);
    h = engine::FnvU64(h, r.pair.residual_flat_zero ? 1u : 0u);
    for (const hecke_ext::LesMapSpot& sp : r.pair.les_maps) {
        h = engine::FnvBytes(h, sp.term);
        h = engine::FnvU64(h, static_cast<uint64_t>(static_cast<int64_t>(sp.q)));
        h = engine::FnvU64(h, sp.dim_middle);
        h = engine::FnvU64(h, sp.rank_in);
        h = engine::FnvU64(h, sp.rank_out);
        h = engine::FnvU64(h, sp.asserted ? (sp.exact ? 2u : 1u) : 0u);
    }
    return h;
}

// ===========================================================================
// Arm 11 -- the schema bump itself, and the record-level `u_ell`.
//
// The ruling ([OD] slot-4 close-out 5) is a statement about a VALUE, not about
// a key: `resumable_items` used to multiply a cell-level refusal by the ideal
// count.  This arm builds the four-ideal cell that shows it.
// ===========================================================================
void RunSchemaBumpArm() {
    std::cout << "-- arm 11: hecke_ext/v2 -- the schema key, and the CELL's u_ell\n";

    // (1) v1 and v2 are distinguishable by the schema key ALONE.
    Check(std::string(hecke_ext::kExtSchemaV1) != std::string(hecke_ext::kExtSchemaV2),
          "the two schema strings differ");
    CheckEq(std::string(hecke_ext::kExtSchema), std::string("hecke_ext/v5"),
            "this build writes hecke_ext/v5 (slot-13 WP-7; v4 since slot-12 FIN; v3 since slot-7 WP-3 stage 3)");
    {
        HeckeExtRecord r;
        r.n_sym = 10; r.ell = 691; r.weight = 12;
        r.bulk.available = true; r.bulk.num_degrees = 3;
        const std::string j = hecke_ext::BuildHeckeExtReportJson(r);
        Check(j.compare(0, 26, "{\"schema\":\"hecke_ext/v5\"") == 0 ||
                  j.find("\"schema\":\"hecke_ext/v5\"") != std::string::npos,
              "the emitted schema string is v2");
        Check(j.find("\"schema\":\"hecke_ext/v1\"") == std::string::npos,
              "and v1 appears nowhere in the text");
        HeckeExtRecord t = r;
        t.schema = hecke_ext::kExtSchemaV1;
        CheckEq(t.verify(), 1u, "verify() rejects a v1 record");
        bool threw = false;
        try { hecke_ext::BuildHeckeExtReportJson(t); }
        catch (const std::exception&) { threw = true; }
        Check(threw, "the writer refuses a v1 record");
    }

    // (2) THE ARM THAT PROVES THE REPAIR.  A four-ideal cell with U_ell
    // refused emits ONE `u_ell` object and counts ONE incomplete item -- where
    // v1 emitted four and counted four ([E] sec:schema v1.5.0's own statement
    // of the defect: "a four-ideal cell emits one decision four times and
    // resumable_items counts it four times").
    HeckeExtRecord r;
    r.n_sym = 16; r.ell = 43867; r.weight = 18;
    r.bulk.available = true; r.bulk.num_degrees = 3;
    r.u_ell.present = true;
    r.u_ell.included = false;
    r.u_ell.reason = "u_ell_over_budget";
    r.u_ell.predicted_mb = 8254.76;
    r.u_ell.budget_mb = 2048;
    r.u_ell.model_version = "u_ell_rss/v2-2026-09-17";
    r.u_ell.resumable.present = true;
    r.u_ell.resumable.kind = hecke_ext::kKindUEll;
    r.u_ell.resumable.complete = false;
    r.u_ell.resumable.budget_json = "{\"budget_mb\":2048}";
    r.u_ell.resumable.budget_history_json = "[2048]";
    for (uint32_t i = 0; i < 4; ++i) {
        IdealRecord id;
        id.id = i;
        id.dim = 1 + i;
        id.local = true;
        id.final_inputs.local = true;
        id.final_inputs.certificates_ok = true;
        id.final_inputs.cyclic_passed = true;
        id.final = hecke_ext::is_final(id.final_inputs, r.settings);
        hecke_ext::IdealGroupMatch own;
        own.group = "H1";
        own.own = true;
        own.matched = true;
        own.ideal_id = i;
        own.witness_poly = {24, 1};
        own.w_hat_exponent = 1;
        own.w_hat_exponent_other = 1;
        id.groups.push_back(own);
        r.ideals.push_back(id);
    }
    CheckEq(r.verify(), 0u, "the four-ideal record verifies");
    CheckEq(r.resumable_items(), 1u,
            "ONE refusal counts ONCE, against v1's four ([OD] slot-4 close-out 5)");
    CheckEq(static_cast<uint32_t>(r.all_resumables().size()), 1u,
            "the record carries exactly one resumable of its own");
    for (const IdealRecord& id : r.ideals) {
        CheckEq(static_cast<uint32_t>(id.all_resumables().size()), 0u,
                "and no ideal carries a u_ell resumable");
    }
    const std::string j = hecke_ext::BuildHeckeExtReportJson(r);
    size_t occurrences = 0;
    for (size_t at = j.find("\"u_ell\":{\"included\""); at != std::string::npos;
         at = j.find("\"u_ell\":{\"included\"", at + 1)) {
        ++occurrences;
    }
    CheckEq(static_cast<uint32_t>(occurrences), 1u,
            "the emitted text carries the verdict exactly once");
    Check(j.find("\"resumable_items\":1") != std::string::npos,
          "and reports resumable_items = 1");

    // An INCLUDED verdict has nothing to resume.
    {
        HeckeExtRecord t = r;
        t.u_ell.included = true;
        CheckEq(t.verify(), 20u, "verify() rejects a resumable on an included verdict");
    }
    {
        HeckeExtRecord t = r;
        t.u_ell.reason = "because";
        CheckEq(t.verify(), 20u, "verify() rejects an unknown u_ell reason");
    }
    {
        HeckeExtRecord t;
        t.n_sym = 10; t.ell = 691; t.weight = 12;
        t.bulk.available = true; t.bulk.num_degrees = 3;
        const std::string tj = hecke_ext::BuildHeckeExtReportJson(t);
        Check(tj.find("\"u_ell\":{\"status\":\"unavailable\"}") != std::string::npos,
              "a cell that formed no verdict emits the placeholder");
    }
}

// ===========================================================================
// Arm 12 -- the per-ideal `groups` list ([E] sec:schema v1.5.0's second repair).
// ===========================================================================
void RunIdealGroupsArm() {
    std::cout << "-- arm 12: the per-ideal `groups` list and its matching witness\n";

    HeckeExtRecord r;
    r.n_sym = 10; r.ell = 691; r.weight = 12;
    r.bulk.available = true; r.bulk.num_degrees = 3;

    IdealRecord a;
    a.id = 0;
    a.dim = 3;
    a.local = true;
    a.final_inputs.local = true;
    a.final_inputs.certificates_ok = true;
    a.final_inputs.cyclic_passed = true;
    a.final = hecke_ext::is_final(a.final_inputs, r.settings);
    hecke_ext::IdealGroupMatch own;
    own.group = "H1"; own.own = true; own.matched = true; own.ideal_id = 0;
    own.witness_poly = {24, 1}; own.w_hat_exponent = 1; own.w_hat_exponent_other = 1;
    a.groups.push_back(own);
    r.ideals.push_back(a);

    IdealRecord b = a;
    b.id = 1;
    b.groups.clear();
    hecke_ext::IdealGroupMatch own2 = own;
    own2.group = "H2"; own2.ideal_id = 1;
    b.groups.push_back(own2);
    hecke_ext::IdealGroupMatch matched;
    matched.group = "H1"; matched.own = false; matched.matched = true; matched.ideal_id = 0;
    matched.witness_poly = {24, 1}; matched.w_hat_exponent = 1; matched.w_hat_exponent_other = 1;
    b.groups.push_back(matched);
    r.ideals.push_back(b);

    CheckEq(r.verify(), 0u, "a record with a cross-group match verifies");
    const std::string j = hecke_ext::BuildHeckeExtReportJson(r);
    Check(j.find("\"groups\":[{\"group\":\"H1\",\"own\":true,\"matched\":true") !=
              std::string::npos,
          "the own-group entry is the identity match");
    Check(j.find("\"witness_poly\":[24,1],\"w_hat_exponent\":1,"
                 "\"w_hat_exponent_other\":1,\"failure\":0") != std::string::npos,
          "and carries the witness the evaluation test used ([XA M5])");
    // `modules` keeps its "one entry per ambient group" shape ([E]).
    Check(j.find("\"modules\":{}") != std::string::npos,
          "`modules` is untouched by the addition");

    // A claimed match with no witness is exactly the key [E] asked for, left
    // unfilled -- refused.
    {
        HeckeExtRecord t = r;
        t.ideals[1].groups[1].witness_poly.clear();
        CheckEq(t.verify(), 21u, "verify() rejects a claimed match with no witness");
    }
    {
        HeckeExtRecord t = r;
        t.ideals[1].groups[1].failure = 2;   // matched AND failed
        CheckEq(t.verify(), 21u, "verify() rejects a match that also reports a failure");
    }
    {
        HeckeExtRecord t = r;
        t.ideals[1].groups[1].matched = false;   // no match and no reason
        CheckEq(t.verify(), 21u, "verify() rejects a non-match with no failure code");
    }
    {
        HeckeExtRecord t = r;
        t.ideals[0].sign = "plus";               // but the record is "all"
        CheckEq(t.verify(), 16u, "verify() rejects an ideal whose sign is not the record's");
    }
}

// ===========================================================================
// Arm 13 -- the complex objects: boundary and pair, `les_maps`, the two
// recorded cone flags, and the placeholder rule they all obey.
// ===========================================================================
void RunComplexObjectsArm() {
    std::cout << "-- arm 13: complexes.boundary / complexes.pair, les_maps, the cone flags\n";

    HeckeExtRecord r;
    r.n_sym = 10; r.ell = 691; r.weight = 12;
    r.bulk.available = true; r.bulk.num_degrees = 3;

    // The shipped model's constants ([C] sec:recordschema item 5: "on the
    // model complexes every field is a constant").
    r.boundary.available = true;
    r.boundary.num_degrees = 2;             // q_max = 1
    r.boundary.model = "circle";
    r.boundary.has_group_cohomology_flag = true;
    r.boundary.is_group_cohomology = true;  // ell != 2

    r.pair.available = true;
    r.pair.num_degrees = 3;                 // q_max = 2, from bulk 2 and boundary 1
    r.pair.derived_from.push_back("bulk");
    r.pair.derived_from.push_back("boundary");
    hecke_ext::LesMapSpot sp;
    sp.term = "Hc"; sp.q = 1; sp.in_built = true; sp.out_built = true; sp.asserted = true;
    sp.dim_middle = 3; sp.rank_in = 1; sp.rank_out = 2;
    sp.composite_zero = true; sp.exact = true; sp.coker_in_dim = 2;
    r.pair.les_maps.push_back(sp);
    hecke_ext::LesMapSpot open_spot;
    open_spot.term = "H_bd"; open_spot.q = 1; open_spot.in_built = true;
    open_spot.out_built = false; open_spot.asserted = false;
    open_spot.dim_middle = 1; open_spot.rank_in = 1; open_spot.coker_in_dim = 0;
    r.pair.les_maps.push_back(open_spot);
    r.pair.has_cone_flags = true;
    r.pair.phi_flat_identity = false;       // recorded, and a `false` IS conformance
    r.pair.residual_flat_zero = false;
    r.pair.action_record_schema = "hecke_cone_action/v1";

    r.les_available = true;
    r.les_exact = true;
    r.les_ranks.emplace_back("r0", 0u);
    r.les_ranks.emplace_back("r1", 1u);
    r.has_dim_H1_shriek = true;
    r.dim_H1_shriek = 2;
    r.groups.push_back({"H1", true, 3});
    r.groups.push_back({"H0_bd", true, 1});
    r.groups.push_back({"H1_bd", true, 1});
    r.groups.push_back({"H1c", true, 3});
    r.groups.push_back({"H2c", true, 0});

    CheckEq(r.verify(), 0u, "a record about three complexes verifies");
    const std::string j = hecke_ext::BuildHeckeExtReportJson(r);
    Check(j.find("\"boundary\":{\"status\":\"computed\",\"q_max\":1,\"truncated\":false,"
                 "\"q_max_source\":\"derived\",\"degrees\":[0,1]") != std::string::npos,
          "the boundary's own four fields (1 / false / derived / [0,1])");
    Check(j.find("\"model\":\"circle\",\"is_group_cohomology\":true") != std::string::npos,
          "and the two labels Hecke/SL2ZBoundaryPair supplied");
    Check(j.find("\"pair\":{\"status\":\"computed\",\"q_max\":2,\"truncated\":false,"
                 "\"q_max_source\":\"derived\",\"degrees\":[0,1,2]") != std::string::npos,
          "the pair's own four fields (2 / false / derived / [0,1,2])");
    Check(j.find("\"derived_from\":[\"bulk\",\"boundary\"]") != std::string::npos,
          "a complex built from others names them ([C] sec:recordschema item 5)");
    Check(j.find("\"les_maps\":[{\"term\":\"Hc\",\"q\":1") != std::string::npos,
          "les_maps carries one object per reported spot ([X] Sec. 6.2 E5)");
    Check(j.find("\"composite_zero\":true,\"exact\":true,\"coker_in_dim\":2") !=
              std::string::npos,
          "an asserted spot reports both readings");
    Check(j.find("\"composite_zero\":null,\"exact\":null") != std::string::npos,
          "an unasserted spot reports `null`, never a default `false`");
    Check(j.find("\"phi_flat_identity\":false,\"residual_flat_zero\":false") !=
              std::string::npos,
          "the two RESERVED cone flags are emitted, and a `false` is conformance");
    Check(j.find("\"action_record\":\"hecke_cone_action/v1\"") != std::string::npos,
          "with the cross-reference to the action's own standalone record");
    Check(j.find("\"dim_H1_shriek\":2") != std::string::npos,
          "the fourth scalar of [E]'s closed list");
    Check(j.find("\"les\":{\"exact\":true,\"hecke_equivariant\":false,"
                 "\"ranks\":{\"r0\":0,\"r1\":1}}") != std::string::npos,
          "and the record-level `les` summary");

    // [C] sec:recordschema item 6(c): an unavailable complex is the
    // placeholder AND NOTHING ELSE -- the v2 content included.
    {
        HeckeExtRecord t = r;
        t.pair.available = false;
        CheckEq(t.verify(), 18u,
                "verify() rejects v2 content on an UNAVAILABLE complex (item 6(c))");
    }
    {
        HeckeExtRecord t = r;
        t.boundary = hecke_ext::ComplexRecord();   // unavailable again
        CheckEq(t.verify(), 19u,
                "verify() rejects dim_H1_shriek without its boundary complex");
        const std::string tj =
                hecke_ext::BuildHeckeExtReportJson([&] { HeckeExtRecord u = t;
                                                          u.has_dim_H1_shriek = false;
                                                          return u; }());
        Check(tj.find("\"dim_H1_shriek\":null") != std::string::npos,
              "and emits it as `null` rather than dropping the key");
    }
    {
        // [C] v1.7.0 `sec:recordschema` item 5 (slot-5 WP-11's ruling, Option
        // A -- docs/implementation/slot5/WP-11_report.md Sec. 3): the names a
        // derived complex carries in `derived_from` ARE the record's own
        // derived flag, so `q_max_source = "set"` together with a non-empty
        // `derived_from` is non-conforming and `verify()` refuses it (code
        // 22).  `t.pair` already carries `derived_from = [bulk, boundary]`
        // from the setup above.  `truncated = true` keeps this arm OFF the
        // UNRELATED code-4 refusal ("set" on a complex that is not
        // truncated, arm 4) -- the trap WP-4 found elsewhere in this
        // codebase, where an earlier shape gate returns before the intended
        // item runs -- so the check below pins that the new code fires
        // SPECIFICALLY for the derived-complex case, not incidentally.
        HeckeExtRecord t = r;
        t.pair.q_max_configured = true;
        t.pair.truncated = true;
        CheckEq(t.verify(), 22u,
                "a DERIVED complex (non-empty derived_from) may not declare "
                "`q_max_source = \"set\"` -- item 5 permits `set` only on a "
                "source complex, and verify() now refuses it by name, not "
                "merely by the unrelated truncation check");
    }
    {
        // The positive control for the SAME clause: the identical `set`
        // declaration on a SOURCE complex (empty `derived_from`) still
        // verifies -- E9's truncated model is untouched by this ruling.
        // (`RunComplexFieldsArm`'s "a declared truncation verifies" arm pins
        // the same positive case independently, on `bulk`.)
        HeckeExtRecord t = r;
        t.boundary.q_max_configured = true;
        t.boundary.truncated = true;
        Check(t.boundary.derived_from.empty(),
              "sanity: the boundary is a SOURCE complex (empty derived_from)");
        CheckEq(t.verify(), 0u,
                "the same `set` declaration on a SOURCE complex still "
                "verifies -- a conforming record is unaffected");
    }
}

// ===========================================================================
// Arm 14 -- the whole v2 record through a from-scratch mini parser.
//
// The `HeckeSimultaneousValidation` pattern: a second, independent reader.  It
// is deliberately not a JSON library -- it is 60 lines that walk the text and
// check that every v2 key it expects is there, nested where the schema says,
// and that the braces and brackets balance.
// ===========================================================================
struct MiniParser {
    const std::string& s;
    size_t i = 0;
    explicit MiniParser(const std::string& t) : s(t) {}
    bool balanced() const {
        long braces = 0, brackets = 0;
        bool in_string = false;
        for (size_t k = 0; k < s.size(); ++k) {
            const char c = s[k];
            if (in_string) { if (c == '"') in_string = false; continue; }
            if (c == '"') { in_string = true; continue; }
            if (c == '{') ++braces;
            else if (c == '}') --braces;
            else if (c == '[') ++brackets;
            else if (c == ']') --brackets;
            if (braces < 0 || brackets < 0) return false;
        }
        return braces == 0 && brackets == 0 && !in_string;
    }
    // The substring of the object that `"<key>":` opens, brace-matched.
    std::string object(const std::string& key) const {
        const size_t at = s.find("\"" + key + "\":");
        if (at == std::string::npos) return std::string();
        size_t k = s.find_first_of("{[", at);
        if (k == std::string::npos) return std::string();
        const char open = s[k], close = (open == '{') ? '}' : ']';
        long depth = 0;
        bool in_string = false;
        for (size_t e = k; e < s.size(); ++e) {
            const char c = s[e];
            if (in_string) { if (c == '"') in_string = false; continue; }
            if (c == '"') { in_string = true; continue; }
            if (c == open) ++depth;
            else if (c == close) { if (--depth == 0) return s.substr(k, e - k + 1); }
        }
        return std::string();
    }
};

void RunRoundTripArm(const HeckeExtRecord& r, const std::string& where) {
    const std::string j = hecke_ext::BuildHeckeExtReportJson(r);
    MiniParser p(j);
    Check(p.balanced(), where + ": the emitted record is balanced");
    // Every top-level key `hecke_ext/v2` promises.
    static const char* top[] = {"schema", "commit", "model", "iota_sign", "n_sym", "ell",
                                "weight", "sign", "coefficient_ring", "poly_coeff_order",
                                "matrix_entry_order", "settings", "resumable_items",
                                "u_ell", "complexes", "groups", "les", "iota",
                                "iota_exact_comparison",
                                "operators_computed", "ideals", "identities",
                                "certificates_ran", "hashes", "verify"};
    for (const char* k : top) {
        Check(HasKey(j, k), where + ": the top-level key `" + k + "`");
    }
    const std::string complexes = p.object("complexes");
    Check(!complexes.empty(), where + ": `complexes` is an object");
    for (const char* c : {"bulk", "boundary", "pair"}) {
        Check(complexes.find(std::string("\"") + c + "\":") != std::string::npos,
              where + std::string(": `complexes` names `") + c + "`");
    }
    const std::string groups = p.object("groups");
    for (const char* k : {"dim_M_k", "dim_S_k", "dim_dickson", "dim_H1_shriek"}) {
        Check(groups.find(std::string("\"") + k + "\":") != std::string::npos,
              where + std::string(": `groups` carries the closed-list scalar `") + k + "`");
    }
    // The second reader's own count of the record's incomplete items.
    uint32_t by_hand = 0;
    for (size_t at = j.find("\"complete\":false"); at != std::string::npos;
         at = j.find("\"complete\":false", at + 1)) {
        ++by_hand;
    }
    CheckEq(by_hand, r.resumable_items(),
            where + ": the reader's own count agrees with `resumable_items`");
    // The writer is a function of the record.
    Check(hecke_ext::BuildHeckeExtReportJson(r) == j, where + ": the writer is deterministic");
}

// ===========================================================================
// Arm 15 -- the v2 objects on a REAL cell, through HeckeExtDriver.
//
// The constructed arms above pin the schema; this one pins that the shipped
// producers fill it.  `(10, 691)` under `u_ell = off` -- the same cell arm 8
// uses, and for the same reason: U_691 is 104 MB and 4 s, and neither is what
// this arm is about ([E] `cl:ramanujan` at full cost is
// HeckeExtRamanujanValidation's).
// ===========================================================================
void RunRealCellV2Arm() {
    std::cout << "-- arm 15: the v2 objects on the real (10, 691) cell (u_ell = off)\n";

    hecke::UEllSetting off;
    off.mode = hecke::UEllMode::Off;
    const hecke::ScopedUEll guard(off);

    hecke_ext::ExtDriverOptions opts;
    opts.with_cone = true;
    opts.with_sign_split = true;
    hecke_ext::HeckeExtDriver d(10, 691, opts);
    d.run();
    const HeckeExtRecord& r = d.record();

    CheckEq(r.verify(), 0u, "(10,691) v2: the record verifies");
    CheckEq(std::string(r.schema), std::string("hecke_ext/v5"), "(10,691) v2 objects: the schema (v5 since slot-13 WP-7)");

    // --- the complexes -------------------------------------------------
    Check(r.bulk.available && r.boundary.available && r.pair.available,
          "(10,691) v2: all three complexes are computed");
    CheckEq(r.bulk.num_degrees, 3u, "(10,691) v2: the bulk's own degree count");
    CheckEq(r.boundary.num_degrees, 2u, "(10,691) v2: the boundary's own degree count");
    CheckEq(r.pair.num_degrees, 3u, "(10,691) v2: the pair's own degree count");
    Check(!r.bulk.q_max_configured && !r.boundary.q_max_configured &&
              !r.pair.q_max_configured,
          "(10,691) v2: every q_max is DERIVED on the finite model");
    CheckEq(r.boundary.model, std::string("circle"), "(10,691) v2: the boundary model");
    Check(r.boundary.has_group_cohomology_flag && r.boundary.is_group_cohomology,
          "(10,691) v2: the boundary is group cohomology at ell != 2");
    Check(r.pair.derived_from.size() == 2 && r.pair.derived_from[0] == "bulk" &&
              r.pair.derived_from[1] == "boundary",
          "(10,691) v2: the pair names the complexes it is derived from");
    Check(!r.pair.les_maps.empty(), "(10,691) v2: the pair carries les_maps");
    Check(r.les_available, "(10,691) v2: the les summary is available");
    Check(r.les_exact, "(10,691) v2: every asserted spot of the les is exact");

    // [E] sec:tests at (10, 691): dim H^1 = 3, dim H^1_c = 3, dim H^2_c = 0,
    // rank r^1 = 1, so dim H^1_! = 2 ([E] v1.4.0's correction).
    uint32_t dim_H1c = 0, dim_H2c = 0, dim_H1 = 0;
    bool have_H1c = false, have_H2c = false;
    for (const hecke_ext::GroupRecord& g : r.groups) {
        if (g.name == "H1") dim_H1 = g.dim;
        if (g.name == "H1c") { dim_H1c = g.dim; have_H1c = true; }
        if (g.name == "H2c") { dim_H2c = g.dim; have_H2c = true; }
    }
    CheckEq(dim_H1, 3u, "(10,691) v2: dim H^1 = 3");
    Check(have_H1c && have_H2c, "(10,691) v2: the cone groups are reported");
    CheckEq(dim_H1c, 3u, "(10,691) v2: dim H^1_c = 3 ([E] sec:tests)");
    CheckEq(dim_H2c, 0u, "(10,691) v2: dim H^2_c = 0");
    Check(r.has_dim_H1_shriek, "(10,691) v2: dim H^1_! is reported");
    CheckEq(r.dim_H1_shriek, 2u,
            "(10,691) v2: dim H^1_! = 2, so H^1_! != H^1 at a Kummer cell "
            "([E] v1.4.0's correction)");

    // --- the involution -------------------------------------------------
    Check(!r.iota.empty(), "(10,691) v2: the iota objects are reported");
    bool saw_degree_one = false;
    for (const hecke_ext::IotaDegreeRecord& ir : r.iota) {
        CheckEq(ir.status, std::string("split"), "(10,691) v2: 2 in R^x, so the split applies");
        CheckEq(ir.dim_plus + ir.dim_minus, ir.dim_H,
                "(10,691) v2: dim H^q_+ + dim H^q_- = dim H^q ([C] prop:iotasplit(1))");
        Check(ir.commutation_asserted,
              "(10,691) v2: [iota^*, A^(q)] = 0 is ASSERTED at ell >= 5");
        CheckEq(ir.operators_commuting, ir.operators,
                "(10,691) v2: every operator commutes with iota^*");
        if (ir.degree == 1u) {
            saw_degree_one = true;
            CheckEq(ir.dim_plus, 2u, "(10,691) v2: dim H^1_+ = 2");
            CheckEq(ir.dim_minus, 1u, "(10,691) v2: dim H^1_- = 1");
            // [E] D6: the Eisenstein line is DEFINED on the plus side and the
            // unsplit socle is 2-dimensional, so it is "undefined" there.
            CheckEq(ir.eisenstein_status, std::string("defined"),
                    "(10,691) v2: the Eisenstein line is defined on H^1_+");
            CheckEq(ir.eisenstein_dim_socle_over_F, 1u,
                    "(10,691) v2: and is a line over F");
        }
    }
    Check(saw_degree_one, "(10,691) v2: degree 1 carries an iota record");

    // --- [C] `obl:iota`(ii)'s RIDER ([OD] 2026-09-22 Q3) ------------------
    // Until WP-12a the rider was HALF discharged: InvolutionCertificate
    // carried `exact_relation_data` and the per-row `has_exact`, and NO
    // emitted record did, so a reader of the record had to ASSUME the exact
    // comparison had happened -- which is the one thing the rider forbids.
    // These are the record's half.
    Check(r.iota_exact.present,
          "(10,691) v2: the record carries obl:iota(ii)'s exact-comparison object");
    Check(r.iota_exact.ran,
          "(10,691) v2: the driver supplies exact relation data, so the Mat22 half RAN");
    Check(r.iota_exact.relation_rows > 0u,
          "(10,691) v2: the exact-comparison object names the relation rows it walked");
    Check(r.iota_exact.rows_with_exact <= r.iota_exact.relation_rows,
          "(10,691) v2: the exact rows are a subset of the relation rows");
    Check(r.iota_exact.rows_compared.size() <= r.iota_exact.rows_with_exact,
          "(10,691) v2: a compared row carries an exact Mat22");
    Check(!r.iota_exact.rows_compared.empty(),
          "(10,691) v2: the exact comparison ran on at least one row");
    // And it re-derives from the PRODUCTION involution's own certificate,
    // which is the object the rider is about.
    {
        const hecke::HeckeInvolution* inv = d.involution();
        Check(inv != nullptr && inv->computed(),
              "(10,691) v2: the driver's production involution is available");
        if (inv != nullptr && inv->computed()) {
            const hecke::InvolutionCertificate& ic = inv->certificate();
            CheckEq(r.iota_exact.ran, ic.exact_relation_data,
                    "(10,691) v2: `ran` IS the certificate's exact_relation_data");
            CheckEq(r.iota_exact.relation_rows,
                    static_cast<uint32_t>(ic.relations.size()),
                    "(10,691) v2: `relation_rows` IS the certificate's row count");
            uint32_t with_exact = 0, compared = 0;
            for (size_t i = 0; i < ic.relations.size(); ++i) {
                const hecke::InvolutionCertificate::RelationRow& row = ic.relations[i];
                if (!row.has_exact) continue;
                ++with_exact;
                if (!ic.exact_relation_data) continue;
                if (!row.matched || row.match_index >= ic.relations.size()) continue;
                if (!ic.relations[row.match_index].has_exact) continue;
                ++compared;
            }
            CheckEq(r.iota_exact.rows_with_exact, with_exact,
                    "(10,691) v2: `rows_with_exact` re-derives from the certificate");
            CheckEq(static_cast<uint32_t>(r.iota_exact.rows_compared.size()), compared,
                    "(10,691) v2: `rows_compared` re-derives from verify()'s own clause-6 predicate");
        }
    }
    // The EMITTED text says it -- the rider asks for a RECORD, not a field.
    {
        const std::string j = hecke_ext::BuildHeckeExtReportJson(r);
        Check(j.find("\"iota_exact_comparison\":{\"ran\":true") != std::string::npos,
              "(10,691) v2: the emitted record states that the exact comparison happened");
        Check(j.find("\"rows_compared\":") != std::string::npos,
              "(10,691) v2: the emitted record names the rows it ran on");
    }

    // --- the refusal, by its own code, with a positive control ------------
    // POSITIVE CONTROL: the untouched record verifies, so every refusal below
    // is the single corruption's and not a pre-existing failure.
    {
        HeckeExtRecord ctl = r;
        CheckEq(ctl.verify(), 0u, "Q3 refusal: the positive control verifies");

        // (1) THE RIDER ITSELF: iota objects, no exact-comparison object.
        HeckeExtRecord t1 = r;
        t1.iota_exact = hecke_ext::IotaExactComparisonRecord();
        CheckEq(t1.verify(), 23u,
                "Q3 refusal: a record reporting iota WITHOUT the exact-comparison object "
                "is refused by code 23");

        // (2) a record claiming comparisons the exact half never ran.
        HeckeExtRecord t2 = r;
        t2.iota_exact.ran = false;
        CheckEq(t2.verify(), 23u,
                "Q3 refusal: `rows_compared` non-empty with `ran = false` is refused (rule A6)");

        // (3) a compared row index outside the relation list.
        HeckeExtRecord t3 = r;
        Check(!t3.iota_exact.rows_compared.empty(), "Q3 refusal: (3) needs a compared row");
        t3.iota_exact.rows_compared.back() = t3.iota_exact.relation_rows;
        CheckEq(t3.verify(), 23u,
                "Q3 refusal: a compared row index outside the relation list is refused");

        // (4) more exact rows than there are rows.
        HeckeExtRecord t4 = r;
        t4.iota_exact.rows_with_exact = t4.iota_exact.relation_rows + 1u;
        CheckEq(t4.verify(), 23u,
                "Q3 refusal: more exact rows than relation rows is refused");

        // (5) an ABSENT object that nevertheless carries content.
        HeckeExtRecord t5 = r;
        t5.iota.clear();
        t5.iota_exact.present = false;
        CheckEq(t5.verify(), 23u,
                "Q3 refusal: an absent exact-comparison object carrying content is refused");

        // (6) THE BOUNDARY: where the rider does NOT apply -- no iota objects
        // and no exact-comparison object -- the record still verifies, so
        // code 23 is the rider's and not a blanket requirement.
        HeckeExtRecord t6 = r;
        t6.iota.clear();
        t6.iota_exact = hecke_ext::IotaExactComparisonRecord();
        CheckEq(t6.verify(), 0u,
                "Q3 boundary: a record with no involution needs no exact-comparison object");
        Check(hecke_ext::BuildHeckeExtReportJson(t6).find(
                      "\"iota_exact_comparison\":{\"status\":\"unavailable\"}") != std::string::npos,
              "Q3 boundary: and the writer emits the item 6(c) placeholder for it");
    }

    // --- the per-sign records, and E3 acceptance (c) ---------------------
    Check(d.has_sign_records(), "(10,691) v2: the two per-sign records were built");
    const HeckeExtRecord& plus = d.sign_record(true);
    const HeckeExtRecord& minus = d.sign_record(false);
    CheckEq(plus.verify(), 0u, "(10,691) v2: the plus record verifies");
    CheckEq(minus.verify(), 0u, "(10,691) v2: the minus record verifies");
    CheckEq(plus.sign, std::string("plus"), "(10,691) v2: and says which sign it is");
    CheckEq(minus.sign, std::string("minus"), "(10,691) v2: likewise the minus record");
    for (const IdealRecord& id : plus.ideals) {
        CheckEq(id.sign, std::string("plus"), "(10,691) v2: a plus ideal carries its sign");
    }
    uint32_t unsplit_dim = 0, signed_dim = 0;
    for (const IdealRecord& id : r.ideals) unsplit_dim += id.dim;
    for (const IdealRecord& id : plus.ideals) signed_dim += id.dim;
    for (const IdealRecord& id : minus.ideals) signed_dim += id.dim;
    CheckEq(signed_dim, unsplit_dim,
            "(10,691) v2: the unsplit record is the SUM of the two per-sign records "
            "([X] Sec. 6.2 E3 acceptance (c))");
    // [E] Claim 1.8 / sec:tests: cyclic on each sign at (10, 691).
    for (const IdealRecord& id : plus.ideals) {
        CheckEq(id.certificate_routes.a, std::string("cyclic"),
                "(10,691) v2: the plus side is cyclic ([E] sec:tests)");
    }
    for (const IdealRecord& id : minus.ideals) {
        CheckEq(id.certificate_routes.a, std::string("cyclic"),
                "(10,691) v2: the minus side is cyclic");
    }
    // ... and NOT cyclic on the unsplit H^1, which is why it needs sturm(B').
    for (const IdealRecord& id : r.ideals) {
        CheckEq(id.certificate_routes.a, std::string("not_cyclic"),
                "(10,691) v2: the UNSPLIT ideal is not cyclic");
    }
    // The Eisenstein line's CanonicalPosition reaches the plus record.
    bool saw_canonical = false;
    for (const IdealRecord& id : plus.ideals) {
        for (const hecke_ext::CanonicalPosition& cp : id.canonical) {
            saw_canonical = true;
            Check(cp.eisenstein.find("socle") != std::string::npos,
                  "(10,691) v2: the Eisenstein line's CanonicalPosition is the socle");
        }
    }
    Check(saw_canonical, "(10,691) v2: the plus record carries a canonical position");

    // --- the per-ideal matching, re-run HERE by the evaluation test -------
    // The record REPORTS the matching; this checks that what it reports is
    // what SameMaximalIdeal says, on the driver's own algebras.
    uint32_t reported = 0, re_derived = 0;
    for (uint32_t i = 0; i < r.ideals.size(); ++i) {
        for (const hecke_ext::IdealGroupMatch& g : r.ideals[i].groups) {
            if (g.own) continue;
            ++reported;
            const LocalAlgebra& prev = d.local_algebra(g.ideal_id);
            const LocalAlgebra& here = d.local_algebra(i);
            if (!prev.ok || !here.ok || prev.primes != here.primes) continue;
            const bool same = hecke_ext::SameMaximalIdeal(prev, here.t).same_ideal;
            if (same == g.matched) ++re_derived;
        }
    }
    CheckEq(re_derived, reported,
            "(10,691) v2: every reported cross-block match re-derives ([XA M5])");

    // --- the whole record through the second reader ----------------------
    RunRoundTripArm(r, "(10,691) v2");
    RunRoundTripArm(plus, "(10,691) v2 [plus]");

    // --- the collectors, DEFINED AND CALLED HERE ONLY --------------------
    const uint64_t h_iota = CollectIota(r);
    const uint64_t h_bd = CollectBoundaryDims(r);
    const uint64_t h_cone = CollectConeDims(r);
    CheckEq(CollectIota(r), h_iota, "iota^q collectors are stable");
    CheckEq(CollectBoundaryDims(r), h_bd, "the boundary collectors are stable");
    CheckEq(CollectConeDims(r), h_cone, "the cone collectors are stable");
    std::cout << "EXT_RECORD_V2 cell=(10,691) complexes=bulk+boundary+pair"
              << " dim_H1c=" << dim_H1c << " dim_H1_shriek=" << r.dim_H1_shriek
              << " iota_degrees=" << r.iota.size()
              << " iota_exact=" << (r.iota_exact.ran ? "ran" : "not_run")
              << " rows_compared=" << r.iota_exact.rows_compared.size()
              << "/" << r.iota_exact.rows_with_exact
              << "/" << r.iota_exact.relation_rows
              << " plus_ideals=" << plus.ideals.size()
              << " minus_ideals=" << minus.ideals.size()
              << " iota=0x" << HexU64(h_iota)
              << " boundary=0x" << HexU64(h_bd)
              << " cone=0x" << HexU64(h_cone) << std::endl;
}

// ===========================================================================
// Arm 16 -- the [X] Sec. 1.6 collector INVENTORY for the objects v2 adds.
//
// WHY AN INVENTORY AND NOT A NEW SET OF COLLECTORS.  Most of [X] Sec. 1.6's
// tier-A names for E3, E4 and E5 already exist, in the unit that owns the
// object: HeckeSignSplit::golden_objects() emits `charpoly_iota^q`,
// `rank_iota^q`, `dim_Hplus^q`, `dim_Hminus^q` and
// `eisenstein_line_defined`; EquivariantMappingCone::golden_objects() emits
// `dim_Hc^q` and the three LES ranks; HeckeConeAction::golden_objects() emits
// `charpoly_A_c(T_p)^q`, `rank_A_c(T_p)^q` and `trace_A_c(T_p)^q`; and
// HeckeExt/HeckeBoundary::golden_objects() emits `dim_H0_bd`, `dim_H1_bd`,
// `rank_r^q`, `charpoly_A_bd^q` and the chi multiset.  All of them were
// shipped DEFINED AND NEVER EMITTED in slots 3 and 4.
//
// What v2 adds is the RECORD's own view of them -- CollectIota,
// CollectBoundaryDims, CollectConeDims above -- which is what a third party
// holding only the JSON can recompute.  This arm is the checklist WP-3 wires:
// it asserts that the shipped names are present and that the record-level
// collectors see the same numbers, so that a name cannot go missing between
// here and the emission.
//
// EVERY collector named here is called from THIS function and from no other,
// and none of them is reachable from Golden/GoldenHarness.cpp.
// ===========================================================================
void RunCollectorInventoryArm() {
    std::cout << "-- arm 16: the [X] Sec. 1.6 collector inventory for the v2 objects\n";

    hecke::UEllSetting off;
    off.mode = hecke::UEllMode::Off;
    const hecke::ScopedUEll guard(off);

    hecke_ext::ExtDriverOptions opts;
    opts.with_cone = true;
    opts.with_sign_split = true;
    hecke_ext::HeckeExtDriver d(10, 691, opts);
    d.run();

    // --- E3 -------------------------------------------------------------
    const hecke_ext::HeckeSignSplit* split = d.sign_split(1);
    Check(split != nullptr, "inventory: the degree-1 sign split exists");
    if (split != nullptr) {
        std::vector<std::string> names;
        for (const hecke_ext::GoldenObject& g : split->golden_objects()) names.push_back(g.object);
        for (const char* want : {"rank_iota^1", "charpoly_iota^1", "dim_Hplus^1",
                                 "dim_Hminus^1", "eisenstein_line_defined", "iota^1"}) {
            bool found = false;
            for (const std::string& n : names) if (n == want) found = true;
            Check(found, std::string("inventory [E3]: `") + want + "` is collected");
        }
        (void)split->golden_digest();   // defined, never emitted
    }

    // --- E5 -------------------------------------------------------------
    const EquivariantMappingCone* cone = d.cone();
    Check(cone != nullptr, "inventory: the cone exists");
    if (cone != nullptr) {
        std::vector<std::string> names;
        for (const ConeGoldenObject& g : cone->golden_objects()) names.push_back(g.object);
        for (const char* want : {"dim_Hc^1", "rank_r^1"}) {
            bool found = false;
            for (const std::string& n : names) if (n == want) found = true;
            Check(found, std::string("inventory [E5]: `") + want + "` is collected");
        }
        (void)cone->golden_digest();    // defined, never emitted
    }

    // --- the RECORD's own view agrees with the objects -------------------
    const HeckeExtRecord& r = d.record();
    if (split != nullptr && cone != nullptr) {
        uint32_t rec_H1c = 0, rec_plus = 0;
        for (const hecke_ext::GroupRecord& g : r.groups) {
            if (g.name == "H1c") rec_H1c = g.dim;
            if (g.name == "H1_plus") rec_plus = g.dim;
        }
        CheckEq(rec_H1c, cone->dim_Hc(1),
                "inventory: the record's H1c is the cone's dim_Hc(1)");
        CheckEq(rec_plus, split->dim_plus(),
                "inventory: the record's H1_plus is the split's dim_plus()");
        CheckEq(r.les_ranks.size() >= 2 ? r.les_ranks[1].second : 0u, cone->rank_r(1),
                "inventory: the record's les.ranks.r1 is the cone's rank_r(1)");
    }
    // The record-level collectors are a function of the record alone.
    HeckeExtRecord copy = r;
    CheckEq(CollectIota(copy), CollectIota(r), "inventory: CollectIota is a pure function");
    CheckEq(CollectBoundaryDims(copy), CollectBoundaryDims(r),
            "inventory: CollectBoundaryDims is a pure function");
    CheckEq(CollectConeDims(copy), CollectConeDims(r),
            "inventory: CollectConeDims is a pure function");
    // ... and moves when the record does.
    copy.les_exact = !copy.les_exact;
    Check(CollectConeDims(copy) != CollectConeDims(r),
          "inventory: and it MOVES when the record does (les_exact)");
    copy = r;
    if (!copy.iota.empty()) {
        copy.iota[0].dim_plus += 1;
        copy.iota[0].dim_minus -= 1;
        Check(CollectIota(copy) != CollectIota(r),
              "inventory: CollectIota moves with dim_Hplus");
    }
}

// ===========================================================================
// Arm 17 -- (slot-7 WP-3, stage 2 item 3; owner ruling 3a; [E] sec:schema (a))
// the cell-local `proof_range`, its closed source list, `anemic_pool`, and a
// tamper arm per new clause of verify() code 25.
// ===========================================================================
void RunProofRangeArm() {
    std::cout << "-- arm 17: cell-local proof_range / proof_range_source / anemic_pool\n";
    hecke::UEllSetting off;
    off.mode = hecke::UEllMode::Off;
    const hecke::ScopedUEll u_guard(off);

    // (24, 5) with the harness family: ideal 1 is Dickson-type (its anemic
    // system over T equals an H^0 system), so the decided range is false and
    // the RE-RUN Sturm route is `heuristic` -- final false (sturm_heuristic).
    hecke_ext::ExtDriverOptions o;
    o.good_primes = {2, 3, 5};
    hecke_ext::HeckeExtDriver d(24, 5, o);
    d.run();
    const HeckeExtRecord& r = d.record();
    CheckEq(r.verify(), 0u, "(24,5) cell-local: the record verifies");
    Check(r.anemic_pool == hecke_ext::AnemicPool(5), "(24,5): anemic_pool = AnemicPool(5)");
    Check(r.anemic_pool == std::vector<uint64_t>({2, 3, 7, 11, 13, 17, 19, 23, 29, 31}),
          "(24,5): anemic_pool = {q <= 31} \\ {5}");
    uint32_t sturm_ideals = 0, lowered = 0;
    for (const IdealRecord& id : r.ideals) {
        if (!id.sturm.present) continue;
        ++sturm_ideals;
        CheckEq(id.sturm.proof_range_source, std::string(hecke_ext::kProofRangeCellLocal),
                "(24,5): every sturm object is decided at the cell");
        if (!id.sturm.proof_range) {
            ++lowered;
            CheckEq(id.sturm.outcome, std::string("heuristic"),
                    "(24,5): the lowered ideal's sweep RAN with the decided range");
            Check(id.final.value == hecke_ext::Final::False &&
                      id.final.reason == hecke_ext::kReasonSturmHeuristic,
                  "(24,5): the lowered ideal is final = false (sturm_heuristic)");
        }
    }
    Check(sturm_ideals > 0, "(24,5): some ideal reached the Sturm route");
    CheckEq(lowered, 1u, "(24,5): exactly ideal 1 is Dickson-type");
    const std::string j = hecke_ext::BuildHeckeExtReportJson(r);
    Check(j.find("\"proof_range\":false,\"proof_range_source\":\"cell_local\"") !=
              std::string::npos,
          "(24,5): the text carries the decided false and its source");
    Check(j.find("\"anemic_pool\":[2,3,7,11,13,17,19,23,29,31]") != std::string::npos,
          "(24,5): the text carries anemic_pool");
    Check(j.find("dickson_count") == std::string::npos,
          "(24,5): the retired source is emitted nowhere");

    // The caller's value is used unverified and SAYS so (the slot-6 default).
    {
        hecke_ext::ExtDriverOptions oc = o;
        oc.proof_range = true;
        hecke_ext::HeckeExtDriver dc(24, 5, oc);
        dc.run();
        const HeckeExtRecord& rc = dc.record();
        CheckEq(rc.verify(), 0u, "(24,5) caller_input: the record verifies");
        uint32_t caller = 0, finals = 0;
        for (size_t i = 0; i < rc.ideals.size(); ++i) {
            if (!rc.ideals[i].sturm.present) continue;
            CheckEq(rc.ideals[i].sturm.proof_range_source,
                    std::string(hecke_ext::kProofRangeCallerInput),
                    "(24,5) caller_input: the source says caller_input");
            ++caller;
            if (rc.ideals[i].final.value == hecke_ext::Final::True) ++finals;
        }
        CheckEq(caller, sturm_ideals, "(24,5) caller_input: the same sturm objects");
        CheckEq(finals, sturm_ideals, "(24,5) caller_input true: every sturm ideal final "
                                      "(the pre-item-3 default reproduced)");
    }

    // At ell = 691, n < ell + 1: H^0 = 0, nothing to share -- the range holds.
    {
        hecke_ext::HeckeExtDriver d691(24, 691, o);
        d691.run();
        for (const IdealRecord& id : d691.record().ideals) {
            if (!id.sturm.present) continue;
            Check(id.sturm.proof_range_source == hecke_ext::kProofRangeCellLocal &&
                      id.sturm.proof_range,
                  "(24,691): H^0 = 0, so cell_local proof_range = true");
        }
    }

    // verify() code 25, one tamper per clause.
    size_t k = r.ideals.size();
    for (size_t i = 0; i < r.ideals.size(); ++i) {
        if (r.ideals[i].sturm.present) { k = i; break; }
    }
    Check(k < r.ideals.size(), "(24,5): a sturm ideal to tamper with");
    if (k < r.ideals.size()) {
        { HeckeExtRecord t = r; t.ideals[k].sturm.proof_range_source = "dickson_count";
          CheckEq(t.verify(), 25u, "tamper 25a: the retired dickson_count is refused"); }
        { HeckeExtRecord t = r; t.ideals[k].sturm.proof_range_source = "atlas";
          CheckEq(t.verify(), 25u, "tamper 25b: a source outside the closed list"); }
        { HeckeExtRecord t = r; t.ideals[k].sturm.has_proof_range = false;
          CheckEq(t.verify(), 25u, "tamper 25c: null under cell_local"); }
        { HeckeExtRecord t = r;
          t.ideals[k].sturm.proof_range_source = hecke_ext::kProofRangeMethodUndefined;
          t.ideals[k].sturm.has_proof_range = false;
          CheckEq(t.verify(), 25u, "tamper 25d: method_undefined with a recorded B'"); }
        { HeckeExtRecord t = r; t.anemic_pool.pop_back();
          CheckEq(t.verify(), 25u, "tamper 25e: anemic_pool is not AnemicPool(ell)"); }
        { HeckeExtRecord t = r; t.ideals[k].sturm.proof_range = !t.ideals[k].sturm.proof_range;
          CheckEq(t.verify(), 25u,
                  "tamper 25f: a range the Sturm route did not run with (flag flipped)"); }
        { HeckeExtRecord t = r; t.ideals[k].sturm.has_B_prime = false;
          CheckEq(t.verify(), 25u, "tamper 25g: cell_local without a recorded B'"); }
        { HeckeExtRecord t = r;
          t.ideals[k].sturm.proof_range_source = hecke_ext::kProofRangeMethodUndefined;
          t.ideals[k].sturm.has_proof_range = false;
          t.ideals[k].sturm.has_B_prime = false;
          t.ideals[k].final_inputs.sturm_proof_range = false;
          CheckEq(t.verify(), 25u,
                  "tamper 25h: method_undefined must leave the verdict input as before (true)"); }
        { HeckeExtRecord t = r;
          t.ideals[k].sturm.proof_range_source = hecke_ext::kProofRangeMethodUndefined;
          t.ideals[k].sturm.has_proof_range = false;
          t.ideals[k].sturm.has_B_prime = false;
          t.ideals[k].final_inputs.sturm_proof_range = true;
          t.ideals[k].final_inputs.sturm_has_B_prime = false;
          t.ideals[k].final = hecke_ext::is_final(t.ideals[k].final_inputs, t.settings);
          CheckEq(t.verify(), 0u, "control: a well-formed method_undefined verifies");
          const std::string tj = hecke_ext::BuildHeckeExtReportJson(t);
          Check(tj.find("\"proof_range\":null,\"proof_range_source\":\"method_undefined\"") !=
                    std::string::npos,
                "control: method_undefined is emitted as proof_range null"); }
    }
}

// ===========================================================================
// Arm 18 -- (slot-7 WP-3, stage 2 item 4; owner ruling 3b (B); [E] sec:schema
// (b)) Stage 2(3) by default: S_sep(ell) from the in-process atlas.  On the
// golden subgrid's harness settings, the default run against the S_0-only
// run (explicit empty `s_sep`): the ideal partition and every field the
// golden raw objects `ext_ideals` / `ext_certify` / the E3 sign-split objects
// read must agree -- the record text moves (`s_sep`, `operators_computed`, the
// per-ideal operator lists), no raw object does ([WP6-9] Sec. 3).
// ===========================================================================
std::string RawFieldsText(const HeckeExtRecord& r) {
    std::ostringstream o;
    o << "verify=" << r.verify() << " ideals=" << r.ideals.size();
    for (const IdealRecord& id : r.ideals) {
        o << " |" << id.id << "," << id.dim << "," << id.sign << "," << id.residue_degree << ","
          << id.residue_degree_certified << "," << id.local << "," << id.algebra_present << ","
          << id.algebra_dim << "," << id.embdim << "," << id.nilpotency << "," << id.gorenstein
          << ",h[";
        for (uint32_t v : id.hilbert) o << v << ";";
        o << "]," << id.certificate << "," << static_cast<int>(id.final.value) << ","
          << id.final.reason << "," << id.certificate_routes.a << "," << id.certificate_routes.b
          << "," << id.certificate_routes.c << ",m[";
        for (const auto& m : id.modules) o << m.structure.verify() << ";";
        o << "]";
    }
    return o.str();
}

void RunSSepDefaultArm() {
    std::cout << "-- arm 18: S_sep in the default Stage 2(3) (in-process atlas)\n";
    hecke::UEllSetting off;
    off.mode = hecke::UEllMode::Off;
    const hecke::ScopedUEll u_guard(off, hecke::SettingSource::Spec);
    uint32_t moved_raw = 0, cells = 0;
    for (uint32_t ell : {5u, 13u, 691u}) {
        for (uint32_t n : {2u, 4u, 10u, 16u, 22u, 24u}) {
            hecke_ext::ExtDriverOptions o;
            o.good_primes = {2, 3, 5};
            o.with_cone = true;
            o.with_sign_split = true;
            hecke_ext::HeckeExtDriver dflt(n, ell, o);
            dflt.run();
            o.s_sep = std::vector<uint64_t>{};
            hecke_ext::HeckeExtDriver s0(n, ell, o);
            s0.run();
            const HeckeExtRecord& a = dflt.record();
            const HeckeExtRecord& b = s0.record();
            ++cells;
            CheckEq(a.verify(), 0u, "S_sep default: the record verifies");
            const bool atlas = hecke_ext::IsInProcessAtlasEll(ell);
            Check(a.s_sep.computed == atlas, "S_sep default: computed iff an in-process atlas ell");
            if (atlas) {
                Check(a.s_sep.primes == hecke_ext::InProcessAtlasSSep(ell),
                      "S_sep default: primes = the in-process atlas's S_sep");
                for (uint64_t q : a.s_sep.added)
                    Check(std::find(a.operators_computed.begin(), a.operators_computed.end(), q) !=
                              a.operators_computed.end(),
                          "S_sep default: every added prime is an operator computed");
            }
            const std::string ra = RawFieldsText(a), rb = RawFieldsText(b);
            bool sign_same = true;
            for (uint32_t q : dflt.degrees_run()) {
                const hecke_ext::HeckeSignSplit* x = dflt.sign_split(q);
                const hecke_ext::HeckeSignSplit* y = s0.sign_split(q);
                if ((x == nullptr) != (y == nullptr)) { sign_same = false; continue; }
                if (x == nullptr) continue;
                const auto gx = x->golden_objects(), gy = y->golden_objects();
                if (gx.size() != gy.size()) { sign_same = false; continue; }
                for (size_t i = 0; i < gx.size(); ++i)
                    if (gx[i].object != gy[i].object || gx[i].hash != gy[i].hash) sign_same = false;
            }
            const bool same = (ra == rb) && sign_same;
            if (!same) ++moved_raw;
            std::cout << "    SSEP_DEFAULT cell=(" << n << "," << ell << ") added=";
            for (uint64_t q : a.s_sep.added) std::cout << q << ";";
            std::cout << " ops=" << a.operators_computed.size() << "/" << b.operators_computed.size()
                      << " raw_fields_equal=" << (ra == rb ? 1 : 0)
                      << " sign_split_objects_equal=" << (sign_same ? 1 : 0) << "\n";
            if (ra != rb) std::cout << "      default: " << ra << "\n      S_0 only: " << rb << "\n";
            Check(same, "S_sep default: no golden raw-object field moves at (" + std::to_string(n) +
                            "," + std::to_string(ell) + ")");
        }
    }
    std::cout << "    SSEP_DEFAULT cells=" << cells << " raw_moved=" << moved_raw << "\n";
    // Tamper arms, verify() code 26.
    {
        hecke_ext::ExtDriverOptions o;
        o.good_primes = {2, 3, 5};
        hecke_ext::HeckeExtDriver d(10, 13, o);
        d.run();
        const HeckeExtRecord& r = d.record();
        Check(r.s_sep.computed && !r.s_sep.added.empty(), "(10,13): Stage 2(3) added a prime");
        { HeckeExtRecord t = r; t.ell = 17; t.weight = t.n_sym + 2;
          CheckEq(t.verify() == 26u || t.verify() == 25u, true,
                  "tamper 26a: computed at an ell without an in-process atlas"); }
        { HeckeExtRecord t = r; std::reverse(t.s_sep.primes.begin(), t.s_sep.primes.end());
          if (t.s_sep.primes.size() > 1) CheckEq(t.verify(), 26u, "tamper 26b: primes not ascending"); }
        { HeckeExtRecord t = r; t.s_sep.added.push_back(29);
          CheckEq(t.verify(), 26u, "tamper 26c: an added prime outside S_sep"); }
        { HeckeExtRecord t = r; t.operators_computed.erase(std::remove(t.operators_computed.begin(),
                                  t.operators_computed.end(), t.s_sep.added[0]),
                                  t.operators_computed.end());
          CheckEq(t.verify(), 26u, "tamper 26d: an added prime that was not computed"); }
        { HeckeExtRecord t = r; t.s_sep.computed = false;
          CheckEq(t.verify(), 26u, "tamper 26e: atlas_unavailable carrying primes"); }
        const std::string j = hecke_ext::BuildHeckeExtReportJson(r);
        Check(j.find("\"s_sep\":{\"status\":\"computed\",\"primes\":[") != std::string::npos,
              "(10,13): the text carries s_sep computed");
        hecke_ext::HeckeExtDriver d2(10, 691, o);
        d2.run();
        Check(hecke_ext::BuildHeckeExtReportJson(d2.record())
                      .find("\"s_sep\":{\"status\":\"atlas_unavailable\"}") != std::string::npos,
              "(10,691): the text carries s_sep atlas_unavailable");
    }
}

// ===========================================================================
// Arm 19 -- (slot-7 WP-3, stage 2 item 5; [E] sec:schema (b)) the atlas label.
// Oracle: an independent FULL-POOL run (good primes = T, U_ell off, the
// atlas not consulted) read through AtlasIdealCharacters, keyed by
// AtlasOrbitKey -- the slot-6 WP-9 route, which shares no code with the
// driver's per-block label.  Per degree the multisets of (key, dim) agree.
// ===========================================================================
void RunAtlasLabelArm() {
    std::cout << "-- arm 19: atlas labels in the record\n";
    hecke::UEllSetting off;
    off.mode = hecke::UEllMode::Off;
    const hecke::ScopedUEll u_guard(off, hecke::SettingSource::Spec);
    uint32_t labelled = 0;
    const std::vector<std::pair<uint32_t, uint32_t>> cells = {
        {24, 5}, {32, 5}, {10, 13}, {22, 13}, {12, 11}, {8, 7}, {16, 7}};
    for (const auto& c : cells) {
        const uint32_t n = c.first, ell = c.second;
        const std::string tag = "(" + std::to_string(n) + "," + std::to_string(ell) + ")";
        hecke_ext::ExtDriverOptions o;
        o.good_primes = {2, 3, 5};
        o.with_sign_split = true;
        hecke_ext::HeckeExtDriver d(n, ell, o);
        d.run();
        const HeckeExtRecord& r = d.record();
        CheckEq(r.verify(), 0u, tag + " labels: the record verifies");

        hecke_ext::ExtDriverOptions fo;
        fo.good_primes = hecke_ext::AnemicPool(ell);
        fo.run_commutant = false;
        fo.run_sturm = false;
        fo.consult_atlas = false;
        hecke_ext::HeckeExtDriver full(n, ell, fo);
        full.run();
        const std::vector<hecke_ext::AtlasIdealCharacter> chars =
                hecke_ext::AtlasIdealCharacters(full);
        std::vector<std::string> want, got;
        for (const hecke_ext::AtlasIdealCharacter& ch : chars) {
            if (!ch.local) continue;
            hecke_ext::CanonicalField F(ell, ch.r);
            const std::vector<hecke_ext::FieldElt> key = hecke_ext::AtlasOrbitKey(F, ch.orbit[0]);
            std::ostringstream s;
            s << ch.group << ":" << ch.dim << ":";
            for (const auto& e : key) { for (uint32_t v : e) s << v << "."; s << "|"; }
            want.push_back(s.str());
        }
        for (const IdealRecord& id : r.ideals) {
            if (!id.local) {
                Check(!id.label.computed, tag + " labels: a non-local ideal carries no label");
                continue;
            }
            Check(id.label.computed, tag + " labels: every local ideal is labelled");
            ++labelled;
            std::ostringstream s;
            s << id.groups[0].group << ":" << id.dim << ":";
            for (const auto& e : id.label.key) { for (uint32_t v : e) s << v << "."; s << "|"; }
            got.push_back(s.str());
        }
        std::sort(want.begin(), want.end());
        std::sort(got.begin(), got.end());
        Check(want == got, tag + " labels: (group, dim, key) agree with the full-pool atlas route");
        // The per-sign ideals carry the label of their unsplit system.
        if (d.has_sign_records()) {
            for (bool plus : {true, false}) {
                for (const IdealRecord& id : d.sign_record(plus).ideals) {
                    if (!id.local) continue;
                    bool found = false;
                    for (const IdealRecord& u : r.ideals)
                        if (u.label.computed && u.label.key == id.label.key &&
                            u.label.system == id.label.system)
                            found = true;
                    Check(id.label.computed && found,
                          tag + " labels: a per-sign ideal carries an unsplit ideal's label");
                }
                CheckEq(d.sign_record(plus).verify(), 0u, tag + " labels: a per-sign record verifies");
            }
        }
        const std::string j = hecke_ext::BuildHeckeExtReportJson(r);
        Check(j.find("\"label\":{\"status\":\"computed\",\"key\":[") != std::string::npos,
              tag + " labels: the text carries a computed label");
    }
    Check(labelled > 0, "labels: some ideal was labelled");
    // Unavailable at ell = 691; tamper arms, verify() code 27.
    {
        hecke_ext::ExtDriverOptions o;
        o.good_primes = {2, 3, 5};
        hecke_ext::HeckeExtDriver d(10, 691, o);
        d.run();
        const std::string j = hecke_ext::BuildHeckeExtReportJson(d.record());
        Check(j.find("\"label\":{\"status\":\"atlas_unavailable\"}") != std::string::npos &&
                  j.find("\"status\":\"computed\",\"key\"") == std::string::npos,
              "(10,691) labels: atlas_unavailable, nothing computed");
        hecke_ext::HeckeExtDriver d5(24, 5, o);
        d5.run();
        const HeckeExtRecord& r = d5.record();
        size_t k = 0;
        while (k < r.ideals.size() && !r.ideals[k].label.computed) ++k;
        Check(k < r.ideals.size(), "(24,5) labels: a labelled ideal to tamper with");
        if (k < r.ideals.size()) {
            { HeckeExtRecord t = r; t.ideals[k].label.type = "boundary";
              CheckEq(t.verify(), 27u, "tamper 27a: a type outside the closed list"); }
            { HeckeExtRecord t = r; t.ideals[k].label.key.pop_back();
              CheckEq(t.verify(), 27u, "tamper 27b: a key not over anemic_pool"); }
            { HeckeExtRecord t = r; t.ideals[k].label.r += 1;
              CheckEq(t.verify(), 27u, "tamper 27c: key entries not of width r"); }
            { HeckeExtRecord t = r; t.ideals[k].label.has_chi = !t.ideals[k].label.has_chi;
              CheckEq(t.verify(), 27u, "tamper 27d: chi disagrees with the type"); }
            { HeckeExtRecord t = r; t.s_sep = hecke_ext::SSepRecord();
              CheckEq(t.verify(), 27u, "tamper 27e: a label without the atlas (s_sep unavailable)"); }
        }
    }
}

// ===========================================================================
// Arm 20 -- (slot-7 WP-3, stage 2 item 6; [E] sec:schema (c)) the canonical
// split as the record's default report form.  The fragment is re-derived by
// an independent SplitInCanonicalField call (its own CanonicalField) and must
// be BYTE-identical; under split_field_report = own_field no ideal carries
// one; CertifyCanonicalSplit passes on every fragment's data.
// ===========================================================================
void RunCanonicalSplitArm() {
    std::cout << "-- arm 20: the canonical split in the record\n";
    hecke::UEllSetting off;
    off.mode = hecke::UEllMode::Off;
    const hecke::ScopedUEll u_guard(off, hecke::SettingSource::Spec);
    for (const auto& c : std::vector<std::pair<uint32_t, uint32_t>>{{24, 5}, {22, 13}, {24, 691},
                                                                     {10, 691}}) {
        const std::string tag = "(" + std::to_string(c.first) + "," + std::to_string(c.second) + ")";
        hecke_ext::ExtDriverOptions o;
        o.good_primes = {2, 3, 5};
        o.with_sign_split = true;
        hecke_ext::HeckeExtDriver d(c.first, c.second, o);
        d.run();
        const HeckeExtRecord& r = d.record();
        CheckEq(r.verify(), 0u, tag + " split: the record verifies");
        uint32_t n_split = 0;
        for (size_t i = 0; i < r.ideals.size(); ++i) {
            const IdealRecord& id = r.ideals[i];
            Check(id.split_present == id.local, tag + " split: present iff local (canonical)");
            if (!id.split_present) continue;
            ++n_split;
            const hecke_ext::LocalAlgebra& L = d.local_algebra(static_cast<uint32_t>(i));
            const hecke_ext::CanonicalSplit sp =
                    hecke_ext::SplitInCanonicalField(c.second, L.witness_poly, L.primes, L.character);
            Check(sp.ok && sp.to_json() == id.split_fragment,
                  tag + " split: the fragment is the independent split, byte for byte");
            CheckEq(hecke_ext::CertifyCanonicalSplit(sp, L.character), 0u,
                    tag + " split: the split certifies");
        }
        Check(n_split > 0, tag + " split: some ideal carries a split");
        if (d.has_sign_records()) {
            for (bool plus : {true, false})
                for (const IdealRecord& id : d.sign_record(plus).ideals)
                    Check(id.split_present == id.local, tag + " split: per-sign present iff local");
        }
        const std::string j = hecke_ext::BuildHeckeExtReportJson(r);
        Check(j.find("\"split\":{\"schema\":\"hecke_canonical_split/v1\"") != std::string::npos,
              tag + " split: the text carries the verbatim fragment");
        Check(d.timings().canonical_field_s > 0.0 &&
                  d.timings().canonical_field_s <= d.timings().total_s,
              tag + " split: timed inside run() (canonical_field_s <= total_s)");
    }
    // own_field: no split key at all.
    {
        hecke::MathSettings m = hecke::math_settings();
        m.split_field_report = hecke::SplitFieldReport::OwnField;
        m.split_field_report_source = hecke::SettingSource::Spec;
        const hecke::ScopedMathSettings g(m);
        hecke_ext::ExtDriverOptions o;
        o.good_primes = {2, 3, 5};
        hecke_ext::HeckeExtDriver d(24, 5, o);
        d.run();
        CheckEq(d.record().verify(), 0u, "own_field: the record verifies");
        for (const IdealRecord& id : d.record().ideals)
            Check(!id.split_present, "own_field: no ideal carries a split");
        Check(hecke_ext::BuildHeckeExtReportJson(d.record()).find("\"split\":") == std::string::npos,
              "own_field: the text carries no split key");
    }
    // Tamper arms, verify() code 28.
    {
        hecke_ext::ExtDriverOptions o;
        o.good_primes = {2, 3, 5};
        hecke_ext::HeckeExtDriver d(24, 5, o);
        d.run();
        const HeckeExtRecord& r = d.record();
        size_t k = 0;
        while (k < r.ideals.size() && !r.ideals[k].split_present) ++k;
        Check(k < r.ideals.size(), "split: an ideal to tamper with");
        if (k < r.ideals.size()) {
            { HeckeExtRecord t = r; t.ideals[k].split_present = false; t.ideals[k].split_fragment.clear();
              CheckEq(t.verify(), 28u, "tamper 28a: a local ideal without its split under canonical"); }
            { HeckeExtRecord t = r; t.ideals[k].split_ok = false;
              CheckEq(t.verify(), 28u, "tamper 28b: a split with ok = false"); }
            { HeckeExtRecord t = r; t.ideals[k].split_fragment.clear();
              CheckEq(t.verify(), 28u, "tamper 28c: an empty fragment"); }
            { HeckeExtRecord t = r; t.settings.split_field_report = hecke::SplitFieldReport::OwnField;
              const uint32_t v = t.verify();
              Check(v == 28u || v != 0u, "tamper 28d: a split under own_field is refused"); }
        }
    }
}

// ===========================================================================
// Arm 21 -- (slot-7 WP-3, stage 2 item 7; [E] sec:schema (d), pr:eiscusp)
// predictions_mode and the prediction objects.  (10, 691) is Ramanujan's
// cell: 691 | num(B_12 / 12), the Eisenstein ideal chi_{0,11} is predicted
// Eisenstein-cuspidal at k = 12 and confirmed (its H^1 module is non-split,
// Loewy [2, 1]); (20, 131) is the M1 Kummer cell 131 | num(B_22 / 22).
// ===========================================================================
void RunPredictionsArm() {
    std::cout << "-- arm 21: predictions_mode and the prediction objects\n";
    hecke::UEllSetting off;
    off.mode = hecke::UEllMode::Off;
    const hecke::ScopedUEll u_guard(off, hecke::SettingSource::Spec);
    for (const auto& c : std::vector<std::pair<uint32_t, uint32_t>>{{10, 691}, {20, 131}}) {
        const std::string tag = "(" + std::to_string(c.first) + "," + std::to_string(c.second) + ")";
        const uint32_t w = c.first + 2;
        hecke_ext::ExtDriverOptions o;
        o.good_primes = {2, 3, 5};
        hecke_ext::HeckeExtDriver d(c.first, c.second, o);
        d.run();
        const HeckeExtRecord& r = d.record();
        CheckEq(r.verify(), 0u, tag + " predictions: the record verifies");
        Check(!r.predictions.kummer.empty() && r.predictions.kummer[0].k == w &&
                  r.predictions.kummer[0].divides,
              tag + " predictions: the weight's Kummer verdict divides");
        Check(!r.predictions.glover_json.empty(), tag + " predictions: Glover factors attached");
        uint32_t predicted = 0;
        for (const IdealRecord& id : r.ideals) {
            for (size_t i = 0; i < id.predicted.size(); ++i) {
                ++predicted;
                CheckEq(id.predicted[i], "eisenstein_cuspidal:k=" + std::to_string(w),
                        tag + " predictions: the id names the weight");
                CheckEq(static_cast<int>(id.confirmed[i]), 1,
                        tag + " predictions: confirmed (X_4 non-split at the Eisenstein ideal)");
            }
        }
        CheckEq(predicted, 1u, tag + " predictions: exactly one Eisenstein-cuspidal prediction");
        const std::string j = hecke_ext::BuildHeckeExtReportJson(r);
        Check(j.find("\"predictions_mode\":\"annotate\"") != std::string::npos,
              tag + " predictions: predictions_mode = annotate by default");
        Check(j.find("\"predicted\":[\"eisenstein_cuspidal:k=" + std::to_string(w) +
                     "\"],\"confirmed\":[true]") != std::string::npos,
              tag + " predictions: the text carries the prediction and its confirmation");
        Check(j.find("\"glover_factors\":{\"schema\":\"hecke_glover_factors/v1\"") != std::string::npos,
              tag + " predictions: the Glover object is embedded");
    }
    // assert: a confirmed prediction lets the run finish; the record says assert.
    {
        hecke::MathSettings m = hecke::math_settings();
        m.predictions = hecke::PredictionsMode::Assert;
        m.predictions_source = hecke::SettingSource::Spec;
        const hecke::ScopedMathSettings g(m);
        hecke_ext::ExtDriverOptions o;
        o.good_primes = {2, 3, 5};
        hecke_ext::HeckeExtDriver d(10, 691, o);
        d.run();
        CheckEq(d.record().verify(), 0u, "assert: (10,691) runs and verifies");
        Check(hecke_ext::BuildHeckeExtReportJson(d.record()).find("\"predictions_mode\":\"assert\"") !=
                  std::string::npos,
              "assert: predictions_mode = assert is recorded");
    }
    // Tamper arms, verify() code 29.
    {
        hecke_ext::ExtDriverOptions o;
        o.good_primes = {2, 3, 5};
        hecke_ext::HeckeExtDriver d(10, 691, o);
        d.run();
        const HeckeExtRecord& r = d.record();
        size_t k = 0;
        while (k < r.ideals.size() && r.ideals[k].predicted.empty()) ++k;
        Check(k < r.ideals.size(), "predictions: an ideal to tamper with");
        if (k < r.ideals.size()) {
            { HeckeExtRecord t = r; t.ideals[k].predicted[0] = "cusp_cusp:k=12";
              CheckEq(t.verify(), 29u, "tamper 29a: a prediction name outside the closed list"); }
            { HeckeExtRecord t = r; t.ideals[k].confirmed.clear();
              CheckEq(t.verify(), 29u, "tamper 29b: confirmed not index-aligned"); }
            { HeckeExtRecord t = r; t.ideals[k].unpredicted.push_back("H7");
              CheckEq(t.verify(), 29u, "tamper 29c: unpredicted names no ambient group"); }
            { HeckeExtRecord t = r; t.predictions.kummer.erase(t.predictions.kummer.begin());
              if (!t.predictions.kummer.empty())
                  CheckEq(t.verify(), 29u, "tamper 29d: the weight's verdict is missing"); }
            { HeckeExtRecord t = r; t.settings.predictions = hecke::PredictionsMode::Assert;
              t.ideals[k].confirmed[0] = 0;
              const uint32_t v = t.verify();
              Check(v == 29u || v != 0u, "tamper 29e: a refuted prediction under assert"); }
        }
    }
}

// slot-13 WP-7 (hecke_ext/v5; [E] v1.14.0-pre.1 cert:boundary; slot-12 FIN
// audit G2): the (d1) data on a real record -- (26, 5) ideal 3, a census row
// the boundary certificate discharges -- and verify() code 32 on tampered
// copies.
void RunBoundaryDataArm() {
    std::cout << "-- arm v5: boundary_data, (d1) re-derived from the record (verify() code 32)\n";
    hecke_ext::HeckeExtDriver drv(26, 5);
    drv.run();
    const HeckeExtRecord& r = drv.record();
    CheckEq(r.verify(), 0u, "v5 (26,5): the record verifies");
    int k = -1;
    uint32_t with_data = 0;
    for (size_t i = 0; i < r.ideals.size(); ++i) {
        const IdealRecord& id = r.ideals[i];
        Check(id.boundary_data.present == (id.certificate_routes.d1 == "passed"),
              "v5 (26,5): boundary_data present iff d1 = passed (ideal " + std::to_string(i) + ")");
        with_data += id.boundary_data.present;
        if (id.boundary_data.present && k < 0) k = static_cast<int>(i);
    }
    Check(k >= 0, "v5 (26,5): an ideal with d1 = passed");
    if (k < 0) return;
    const IdealRecord& id = r.ideals[k];
    const uint32_t D = id.dim;
    CheckEq(id.boundary_data.rank_r1, D, "v5 (26,5): rank_r1 = D");
    CheckEq(id.boundary_data.r1.cols, D, "v5 (26,5): r1 has D columns");
    CheckEq(id.boundary_data.t.size(), id.boundary_reps.size(), "v5 (26,5): one t_p per boundary_rep");
    CheckEq(hecke_ext::BoundaryDataRederive(5, id.operator_primes, id.t_p, id.boundary_reps,
                                            id.boundary_data.rank_r1, id.boundary_data.r1,
                                            id.boundary_data.t),
            0u, "v5 (26,5): (d1) re-derives from the record's own data");
    const std::string j = hecke_ext::BuildHeckeExtReportJson(r);
    size_t occ = 0;
    for (size_t pos = j.find("\"boundary_data\":"); pos != std::string::npos;
         pos = j.find("\"boundary_data\":", pos + 1)) {
        ++occ;
    }
    CheckEq(occ, static_cast<size_t>(with_data), "v5 (26,5): one boundary_data object per (d1)-passed ideal in the text");
    std::ostringstream want;
    want << "\"boundary_data\":{\"rank_r1\":" << D << ",\"r1\":[";
    Check(j.find(want.str()) != std::string::npos, "v5 (26,5): the text carries rank_r1 and r1");
    Check(j.find("\"t\":{\"11\":[[") != std::string::npos, "v5 (26,5): t is keyed by R(5), class order, from 11");
    auto code = [&](const std::function<void(IdealRecord&)>& f) {
        HeckeExtRecord t = r;
        f(t.ideals[k]);
        return t.verify();
    };
    CheckEq(code([](IdealRecord& x) { x.boundary_data.present = false; }), 32u,
            "tamper 32a: d1 = passed without boundary_data");
    CheckEq(code([](IdealRecord& x) { x.boundary_data.rank_r1 += 1; }), 32u,
            "tamper 32b: rank_r1 misstated");
    CheckEq(code([](IdealRecord& x) {
                for (auto& row : x.boundary_data.r1.data) std::fill(row.begin(), row.end(), 0u);
            }), 32u, "tamper 32c: r^1|_V zeroed (H1 fails)");
    CheckEq(code([](IdealRecord& x) { x.boundary_data.t.pop_back(); }), 32u,
            "tamper 32d: a class's t_p missing");
    CheckEq(code([](IdealRecord& x) { x.boundary_data.t[0].data[0][0] = 5; }), 32u,
            "tamper 32e: an entry outside [0, ell)");
    // R(5) = {11, 2, 3, 19}; t_2 is also a stored generator (2 in S).
    CheckEq(code([](IdealRecord& x) {
                ExtMatrix& m = x.boundary_data.t[1];
                m.data[0][0] = (m.data[0][0] + 1) % 5;
            }), 32u, "tamper 32f: t_2 differs from the stored t_2");
    if (D >= 2) {
        CheckEq(code([&](IdealRecord& x) {
                    ExtMatrix e(D, D, 5);
                    e.data[D - 1][0] = 1;   // a nilpotent E_{D,1}: outside the local A_S or not commuting
                    x.boundary_data.t[3] = e;
                }), 32u, "tamper 32g: t_19 replaced by E_{D,1} (H2 / commutation fails)");
    }
    CheckEq(code([&](IdealRecord& x) {
                (void)x;
            }), 0u, "tamper control: an untouched copy verifies");
    // boundary_data on an ideal whose d1 is not passed.
    for (size_t i = 0; i < r.ideals.size(); ++i) {
        if (r.ideals[i].certificate_routes.d1 == "passed") continue;
        HeckeExtRecord t = r;
        t.ideals[i].boundary_data = id.boundary_data;
        CheckEq(t.verify(), 32u, "tamper 32h: boundary_data on an ideal whose d1 is " +
                                     r.ideals[i].certificate_routes.d1);
        break;
    }
    std::cout << "EXT_RECORD_V5 cell=(26,5) ideal=" << k << " D=" << D << " h1_bd=" << id.boundary_data.r1.rows
              << " reps=" << id.boundary_reps.size() << " d1_passed=" << with_data
              << " json_bytes=" << j.size() << "\n";
}

} // namespace

uint32_t HeckeExtRecordValidation() {
    Check.reset();
    std::cout << "=== HeckeExtRecordValidation (hecke-extensions slot 4 WP-1 / E2a; "
                 "slot 5 WP-1 / E2 cont.) ===\n";
    std::cout << "    the hecke_ext/v2 record, `Final`, the settings fingerprint, "
                 "commutant_budget, the sidecar\n";
    std::cout << "    slot-5 WP-1: the schema bump, the CELL's u_ell, the per-ideal groups, "
                 "the boundary / pair complexes with les_maps, and the per-sign records\n";

    RunFinalTypeArm();
    RunSettingsKeysArm();
    RunIsFinalArm();
    RunComplexFieldsArm();
    RunKeySetArm();
    RunSidecarArm();
    RunCommutantBudgetArm();
    RunRealCellArm();
    RunJobIdArm();
    RunConcurrencyArm();
    // slot-5 WP-1: `hecke_ext/v2`.
    RunSchemaBumpArm();
    RunIdealGroupsArm();
    RunComplexObjectsArm();
    RunRealCellV2Arm();
    RunCollectorInventoryArm();
    // slot-7 WP-3: stage 2 item 3.
    RunProofRangeArm();
    // slot-7 WP-3: stage 2 item 4.
    RunSSepDefaultArm();
    // slot-7 WP-3: stage 2 item 5.
    RunAtlasLabelArm();
    // slot-7 WP-3: stage 2 item 6.
    RunCanonicalSplitArm();
    // slot-7 WP-3: stage 2 item 7.
    RunPredictionsArm();
    // slot-13 WP-7: hecke_ext/v5, (d1)'s data and verify() code 32.
    RunBoundaryDataArm();

    std::cout << "EXT_RECORD checks=" << Check.checks << " failures=" << Check.fails << std::endl;
    if (Check.fails == 0) {
        std::cout << "=== HeckeExtRecordValidation PASSED (" << Check.checks << " checks) ===\n";
    } else {
        std::cout << "=== HeckeExtRecordValidation FAILED (" << Check.fails << " of " << Check.checks
                  << " checks) ===\n";
    }
    return Check.fails;
}
