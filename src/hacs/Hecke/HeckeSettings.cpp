// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// (hecke-extensions phase E0b, slot 2 WP-1) The mathematical settings layer.
// See HeckeSettings.h for the contract; [X] Sec. 1.2 / Sec. 1.5, [OD] R1.

#include "HeckeSettings.h"

#include "../../ModArithmetic/Fnv.h"
#include "../../Engine/EngineRuntime.h"

#include <cstdlib>
#include <sstream>

namespace hecke {

const char* const kNotImplementedInThisBuildTag = "not_implemented_in_this_build:";

const char* to_string(SettingSource s) {
    switch (s) {
        case SettingSource::Default: return "default";
        case SettingSource::Spec:    return "spec";
        case SettingSource::Console: return "console";
    }
    return "default";
}

// ---------------------------------------------------------------------------
// The registry.  ONE list, in the order canonical_text() / to_json() use; the
// fingerprint is a function of this order, so a key is APPENDED, never
// inserted, once a value has been published under it.
// ---------------------------------------------------------------------------
namespace {
const std::vector<std::string>& registry() {
    static const std::vector<std::string> k = {
        "locality",
        "u_ell",  // slot-3 WP-1 (E1b-0); consumed by Stage 1 in WP-2 (E1b)
        // slot-3 WP-2 (E1b).  The [X] Sec. 1.5 rows `sturm.cap` and
        // `sturm.unavailable`, under that table's campaign-spec spellings.
        // Consumer: HeckeExt/HeckeSturmSweep (certificate (c)).
        "sturm",
        "sturm_unavailable",
        // slot-4 WP-1 (E2a).  [X] Sec. 1.5's "the commutant's kCommutantBudget
        // (a named constant today, a setting only if a consumer asks)" and
        // [OD] slot-3 confirmation 3 ("kCommutantBudget as a setting").  The
        // consumer is R3.1 item 4: a resume with a raised budget is a NEW
        // RECORDED SETTING, and E2's record carries `resumable.budget`.
        // Reader: HeckeExt/HeckeGenerationCertificates' CommutantRoute.
        "commutant_budget",
        // slot-4 WP-1 (E2a).  [X] Sec. 1.5 table row 1 / Sec. 1.3.1 R3.2; its
        // consumer, hecke_ext::is_final (HeckeExt/HeckeExtRecord), ships in
        // the SAME package, which is what [X] Sec. 1.5's "Consumer (phase)"
        // column asks for ("is_final (E2)").
        "final_mode",
        // slot-5 WP-7 (E6).  [X] Sec. 1.5 table rows `field_polynomial` (OQ9)
        // and `split_field_report` (OQ10), [X] Sec. 2.2 Stage 6.  The first
        // has its consumer in the SAME package
        // (HeckeExt/HeckeCanonicalField's FieldPolynomialFromSettings, which
        // raises NotImplementedInThisBuild on `conway`); the second is the
        // `u_ell` pattern of slot-3 WP-1 -- the key, its parser and its
        // validation ship one slot before the REPORT layer that reads it,
        // which is E6's, in slot 6.  Both default, so the all-defaults
        // fingerprint is still 0.
        "field_polynomial",
        "split_field_report",
        // slot-7 WP-2 ([X] Sec. 8 stage 1 (2), [X] Sec. 1.5 rows `presentation`
        // (OQ11) and `predictions` (OQ14)).  Registered AFTER stage 1 (1)'s
        // path-scoped golden recipe, which makes a key registered at its
        // default golden-additive; both default, so the all-defaults
        // fingerprint is still 0.  `presentation`'s consumer ships with it
        // (HeckeExt/HeckeExtDriver: the per-ideal record object, present only
        // under `on`); `predictions` is the `split_field_report` precedent --
        // its consumer, the record's `predictions_mode`, is slot-7 WP-3 item 7.
        "presentation",
        "predictions",
        // slot-8 WP-3 (E8 item 3; [E] v1.9.0-pre.1 sec:ring (a); slot-8
        // decision 4 (A)).  Appended at the tail: eleven keys.  At its default
        // (`F_ell`) it contributes nothing to the fingerprint, which stays 0;
        // a non-default value is honoured by EquivariantCohomologyOnK and
        // HeckeOperatorOnK only, and refused by name everywhere else
        // (Hecke/CoefficientRing.h).
        "coefficient_ring",
        // The remaining keys of [X] Sec. 1.5 arrive with their consuming
        // phases -- APPENDED here, never inserted, once a value has been
        // published under a key (the fingerprint is a function of this order).
    };
    return k;
}
} // namespace

const std::vector<std::string>& MathSettings::keys() { return registry(); }

bool MathSettings::known_key(const std::string& key) {
    for (const std::string& k : registry()) {
        if (k == key) return true;
    }
    return false;
}

// ---------------------------------------------------------------------------
// `locality`
// ---------------------------------------------------------------------------

std::string LocalitySetting::to_text() const {
    if (mode == LocalityMode::Strict) return "strict";
    std::ostringstream oss;
    oss << "budget=" << budget;
    return oss.str();
}

LocalitySetting ParseLocality(const std::string& value) {
    if (value == "strict") {
        // LEGAL, and not degraded: the consumer raises
        // NotImplementedInThisBuild (see HeckeSettings.h).  The budget member
        // keeps its default, which strict mode reads as the linear phase's
        // budget ([X] Sec. 3.1 step 4).
        LocalitySetting s;
        s.mode = LocalityMode::Strict;
        s.budget = kDefaultLocalityBudget;
        return s;
    }
    const std::string kPrefix = "budget=";
    if (value.size() > kPrefix.size() && value.compare(0, kPrefix.size(), kPrefix) == 0) {
        const std::string digits = value.substr(kPrefix.size());
        for (char c : digits) {
            if (c < '0' || c > '9') {
                throw std::invalid_argument(
                    "HeckeSettings: locality = budget=<N> needs a decimal N (got \"" + value + "\")");
            }
        }
        if (digits.size() > 9) {
            throw std::invalid_argument(
                "HeckeSettings: locality budget out of range (got \"" + value + "\")");
        }
        const uint32_t n = static_cast<uint32_t>(std::strtoul(digits.c_str(), nullptr, 10));
        if (n == 0) {
            throw std::invalid_argument(
                "HeckeSettings: locality = budget=0 would consume no witness candidate at all; "
                "use budget=1 for the tightest bounded-exit run");
        }
        LocalitySetting s;
        s.mode = LocalityMode::Budget;
        s.budget = n;
        return s;
    }
    throw std::invalid_argument(
        "HeckeSettings: unknown locality value \"" + value + "\" (expected budget=<N> or strict)");
}

// ---------------------------------------------------------------------------
// `u_ell` (slot-3 WP-1; see HeckeSettings.h)
// ---------------------------------------------------------------------------

std::string UEllSetting::to_text() const {
    switch (mode) {
        case UEllMode::Off: return "off";
        case UEllMode::BudgetMb: {
            std::ostringstream oss;
            oss << "budget_mb=" << budget_mb;
            return oss.str();
        }
        case UEllMode::Auto:
        default:
            return "auto";
    }
}

UEllSetting ParseUEll(const std::string& value) {
    if (value == "auto") return UEllSetting();
    if (value == "off") {
        UEllSetting s;
        s.mode = UEllMode::Off;
        return s;
    }
    const std::string kPrefix = "budget_mb=";
    if (value.size() > kPrefix.size() && value.compare(0, kPrefix.size(), kPrefix) == 0) {
        const std::string digits = value.substr(kPrefix.size());
        for (char c : digits) {
            if (c < '0' || c > '9') {
                throw std::invalid_argument(
                    "HeckeSettings: u_ell = budget_mb=<M> needs a decimal M in MB (got \"" +
                    value + "\")");
            }
        }
        if (digits.size() > 9) {
            throw std::invalid_argument(
                "HeckeSettings: u_ell budget out of range (got \"" + value + "\")");
        }
        const uint32_t m = static_cast<uint32_t>(std::strtoul(digits.c_str(), nullptr, 10));
        if (m == 0) {
            throw std::invalid_argument(
                "HeckeSettings: u_ell = budget_mb=0 would refuse every cell; use u_ell = off");
        }
        UEllSetting s;
        s.mode = UEllMode::BudgetMb;
        s.budget_mb = m;
        return s;
    }
    throw std::invalid_argument(
        "HeckeSettings: unknown u_ell value \"" + value +
        "\" (expected auto, off or budget_mb=<M>)");
}

// ---------------------------------------------------------------------------
// `sturm` / `sturm_unavailable` (slot-3 WP-2; see HeckeSettings.h)
// ---------------------------------------------------------------------------

std::string SturmSetting::to_text() const {
    switch (mode) {
        case SturmCapMode::Off: return "off";
        case SturmCapMode::Cap: {
            std::ostringstream oss;
            oss << "cap=" << cap;
            return oss.str();
        }
        case SturmCapMode::Auto:
        default:
            return "auto";
    }
}

SturmSetting ParseSturm(const std::string& value) {
    if (value == "auto") return SturmSetting();
    if (value == "off") {
        SturmSetting s;
        s.mode = SturmCapMode::Off;
        return s;
    }
    const std::string kPrefix = "cap=";
    if (value.size() > kPrefix.size() && value.compare(0, kPrefix.size(), kPrefix) == 0) {
        const std::string digits = value.substr(kPrefix.size());
        for (char c : digits) {
            if (c < '0' || c > '9') {
                throw std::invalid_argument(
                    "HeckeSettings: sturm = cap=<B> needs a decimal B (got \"" + value + "\")");
            }
        }
        if (digits.size() > 9) {
            throw std::invalid_argument(
                "HeckeSettings: sturm cap out of range (got \"" + value + "\")");
        }
        const uint32_t b = static_cast<uint32_t>(std::strtoul(digits.c_str(), nullptr, 10));
        if (b < 2) {
            throw std::invalid_argument(
                "HeckeSettings: sturm = cap=" + digits +
                " is below 2, the least prime the bound search can stop at; use cap=2 or off");
        }
        SturmSetting s;
        s.mode = SturmCapMode::Cap;
        s.cap = b;
        return s;
    }
    throw std::invalid_argument(
        "HeckeSettings: unknown sturm value \"" + value + "\" (expected auto, off or cap=<B>)");
}

const char* to_text(SturmUnavailablePolicy p) {
    return p == SturmUnavailablePolicy::Heuristic ? "heuristic" : "nonfinal";
}

SturmUnavailablePolicy ParseSturmUnavailable(const std::string& value) {
    if (value == "nonfinal") return SturmUnavailablePolicy::NonFinal;
    if (value == "heuristic") return SturmUnavailablePolicy::Heuristic;
    throw std::invalid_argument(
        "HeckeSettings: unknown sturm_unavailable value \"" + value +
        "\" (expected nonfinal or heuristic)");
}

// ---------------------------------------------------------------------------
// `commutant_budget` (slot-4 WP-1; see HeckeSettings.h)
// ---------------------------------------------------------------------------

std::string CommutantBudgetSetting::to_text() const {
    std::ostringstream oss;
    oss << budget;
    return oss.str();
}

CommutantBudgetSetting ParseCommutantBudget(const std::string& value) {
    if (value.empty()) {
        throw std::invalid_argument(
            "HeckeSettings: commutant_budget = <N> needs a decimal N (got an empty value)");
    }
    for (char c : value) {
        if (c < '0' || c > '9') {
            throw std::invalid_argument(
                "HeckeSettings: commutant_budget = <N> needs a decimal N in modmul (got \"" +
                value + "\")");
        }
    }
    if (value.size() > 19) {
        throw std::invalid_argument(
            "HeckeSettings: commutant_budget out of range (got \"" + value + "\")");
    }
    const uint64_t n = std::strtoull(value.c_str(), nullptr, 10);
    if (n == 0) {
        throw std::invalid_argument(
            "HeckeSettings: commutant_budget = 0 would refuse every commutant system before its "
            "first row; use a positive budget (the route reports `over_budget` on its own)");
    }
    CommutantBudgetSetting s;
    s.budget = n;
    return s;
}

// ---------------------------------------------------------------------------
// `final_mode` (slot-4 WP-1; see HeckeSettings.h)
// ---------------------------------------------------------------------------

const char* to_text(FinalMode m) {
    return m == FinalMode::SturmHeuristic ? "sturm_heuristic" : "closure";
}

FinalMode ParseFinalMode(const std::string& value) {
    if (value == "closure") return FinalMode::Closure;
    if (value == "sturm_heuristic") return FinalMode::SturmHeuristic;
    throw std::invalid_argument(
        "HeckeSettings: unknown final_mode value \"" + value +
        "\" (expected closure or sturm_heuristic)");
}

// ---------------------------------------------------------------------------
// `field_polynomial` / `split_field_report` (slot-5 WP-7; see HeckeSettings.h)
// ---------------------------------------------------------------------------

const char* to_text(FieldPolynomialMode m) {
    return m == FieldPolynomialMode::Conway ? "conway" : "least";
}

FieldPolynomialMode ParseFieldPolynomial(const std::string& value) {
    if (value == "least") return FieldPolynomialMode::Least;
    // LEGAL, and not degraded: the CONSUMER raises NotImplementedInThisBuild
    // (HeckeExt/HeckeCanonicalField's FieldPolynomialFromSettings), exactly as
    // `locality = strict` did between slot 2 and slot 3.
    if (value == "conway") return FieldPolynomialMode::Conway;
    throw std::invalid_argument(
        "HeckeSettings: unknown field_polynomial value \"" + value +
        "\" (expected least or conway)");
}

const char* to_text(SplitFieldReport m) {
    return m == SplitFieldReport::OwnField ? "own_field" : "canonical";
}

SplitFieldReport ParseSplitFieldReport(const std::string& value) {
    if (value == "canonical") return SplitFieldReport::Canonical;
    if (value == "own_field") return SplitFieldReport::OwnField;
    throw std::invalid_argument(
        "HeckeSettings: unknown split_field_report value \"" + value +
        "\" (expected canonical or own_field)");
}

const char* to_text(PresentationMode m) {
    return m == PresentationMode::On ? "on" : "off";
}

PresentationMode ParsePresentation(const std::string& value) {
    if (value == "off") return PresentationMode::Off;
    if (value == "on") return PresentationMode::On;
    throw std::invalid_argument(
        "HeckeSettings: unknown presentation value \"" + value + "\" (expected off or on)");
}

const char* to_text(PredictionsMode m) {
    return m == PredictionsMode::Assert ? "assert" : "annotate";
}

PredictionsMode ParsePredictions(const std::string& value) {
    if (value == "annotate") return PredictionsMode::Annotate;
    if (value == "assert") return PredictionsMode::Assert;
    throw std::invalid_argument(
        "HeckeSettings: unknown predictions value \"" + value +
        "\" (expected annotate or assert)");
}

// ---------------------------------------------------------------------------
// `coefficient_ring` (slot-8 WP-3; see HeckeSettings.h and [E] sec:ring (a))
// ---------------------------------------------------------------------------

std::string CoefficientRingSetting::to_text() const {
    if (k == 0) return "Z";   // (slot-24 D6) un-reserved: the integral descent's ring
    if (k == 1) return "F_ell";
    return "Z/ell^" + std::to_string(k);
}

namespace {
// A non-empty decimal digit string with no sign and no leading zero.
bool plain_decimal(const std::string& t) {
    if (t.empty()) return false;
    for (char ch : t) if (ch < '0' || ch > '9') return false;
    return !(t.size() > 1 && t[0] == '0');
}
} // namespace

CoefficientRingSetting ParseCoefficientRing(const std::string& value) {
    CoefficientRingSetting out;
    if (value == "F_ell") return out;
    static const std::string kPrefix = "Z/ell^";
    if (value.compare(0, kPrefix.size(), kPrefix) == 0) {
        const std::string t = value.substr(kPrefix.size());
        // (slot-11 WP-18c; [E] v1.12.0-pre.2 sec:ring (a)) the parse cap is 4096
        // (it was 63): whether ell^k is executable is decided at the CELL.
        if (plain_decimal(t) && t.size() <= 4) {
            const uint32_t k = static_cast<uint32_t>(std::stoul(t));
            if (k >= 2 && k <= 4096) { out.k = k; return out; }
        }
        throw std::invalid_argument(
            "HeckeSettings: invalid coefficient_ring value \"" + value +
            "\" (expected F_ell or Z/ell^<k> with 2 <= k <= 4096, k in decimal without leading zero; "
            "F_ell has exactly one spelling)");
    }
    // (slot-24 D6, DESCENT_plan.md Sec. 4 D6, OD-10 (a)) `Z` is UN-RESERVED: it is stored (k = 0) and honoured
    // by the integral descent only (Integral/Descent/DescentRecord.h: the record, the console verb `descent`, the
    // campaign's descent cells; [C] thm:descentcert).  Every other construction site refuses it by name through
    // CoefficientRingK(who) / RequireFieldCoefficientRing (exit 5), as it refuses Z/ell^k where it does.
    if (value == "Z") {
        out.k = 0;
        return out;
    }
    // The reserved values ([E] sec:ring (a)): legal, refused HERE, never stored.
    if (value.size() > 2 && value.compare(0, 2, "Z/") == 0) {
        const std::string m = value.substr(2);
        if (plain_decimal(m) && m != "0" && m != "1") {
            throw NotImplementedInThisBuild(
                "coefficient_ring=" + value +
                " (a general Z/m, outside ENGINE_API.md Sec. 13) is reserved and not honoured in this build");
        }
    }
    throw std::invalid_argument(
        "HeckeSettings: unknown coefficient_ring value \"" + value +
        "\" (expected F_ell or Z/ell^<k>, 2 <= k <= 4096)");
}

// ---------------------------------------------------------------------------
// MathSettings
// ---------------------------------------------------------------------------

const MathSettings& MathSettings::defaults() {
    static const MathSettings d;
    return d;
}

// ---------------------------------------------------------------------------
// The per-key VIEW -- the one place a registered key's value is spelled.
// (hecke-extensions slot 5, WP-10 item 2.)
//
// Before this package `canonical_text()`, `to_json()` and `text()` each
// carried their OWN per-key list: three hand-written copies of the same six
// spellings, in the same order, that a seventh key had to be added to three
// times.  `text()`'s own header ([S4] WP-1) recorded the risk and declined to
// take it, on the grounds that the other two are the fingerprint's and the
// record's basis and a refactor would put an emitted string at risk.  This
// package takes it under the one gate that makes it safe: the emitted text is
// asserted byte-unchanged (WP-10 acceptance (c)), on every key and on a
// non-default combination.
//
// The view carries the three things the three writers need -- the canonical
// TEXT, whether the value is AT ITS DEFAULT (which is what canonical_text()
// suppresses), and its SOURCE (which is what to_json() stamps).  The
// at-default test is deliberately the STRUCT comparison the hand-written list
// used, not `text == defaults().text(key)`: a struct may carry a member the
// text does not render (`u_ell = {Auto, budget_mb = 1024}`, `sturm =
// {Auto, cap = 5}`), and the two tests disagree exactly there.  Preserving the
// old test is what makes this package value-preserving by construction rather
// than by inspection.
//
// The final `throw` is the point of the function: a key appended to
// registry() without an arm here fails LOUDLY -- in canonical_text(), in
// to_json(), in text() and therefore in the first console session -- instead
// of silently rendering "" or being dropped from the fingerprint.
namespace {

struct KeyView {
    std::string text;
    bool at_default = true;
    SettingSource source = SettingSource::Default;
};

KeyView view_of(const MathSettings& s, const std::string& key) {
    if (key == "locality") {
        return {s.locality.to_text(), s.locality == LocalitySetting(), s.locality_source};
    }
    if (key == "u_ell") {
        return {s.u_ell.to_text(), s.u_ell == UEllSetting(), s.u_ell_source};
    }
    if (key == "sturm") {
        return {s.sturm.to_text(), s.sturm == SturmSetting(), s.sturm_source};
    }
    if (key == "sturm_unavailable") {
        return {to_text(s.sturm_unavailable),
                s.sturm_unavailable == SturmUnavailablePolicy::NonFinal,
                s.sturm_unavailable_source};
    }
    if (key == "commutant_budget") {
        return {s.commutant_budget.to_text(), s.commutant_budget == CommutantBudgetSetting(),
                s.commutant_budget_source};
    }
    if (key == "final_mode") {
        return {to_text(s.final_mode), s.final_mode == FinalMode::Closure, s.final_mode_source};
    }
    // (slot-5 WP-7 / E6) The two Stage-6 keys, expressed in WP-10's unified
    // view: one arm each, and NOT a third copy in canonical_text() / to_json()
    // / text().  Both at-default tests are the enum comparisons WP-7 wrote.
    if (key == "field_polynomial") {
        return {to_text(s.field_polynomial), s.field_polynomial == FieldPolynomialMode::Least,
                s.field_polynomial_source};
    }
    if (key == "split_field_report") {
        return {to_text(s.split_field_report), s.split_field_report == SplitFieldReport::Canonical,
                s.split_field_report_source};
    }
    // (slot-7 WP-2) The stage-1 (2) keys.
    if (key == "presentation") {
        return {to_text(s.presentation), s.presentation == PresentationMode::Off,
                s.presentation_source};
    }
    if (key == "predictions") {
        return {to_text(s.predictions), s.predictions == PredictionsMode::Annotate,
                s.predictions_source};
    }
    // (slot-8 WP-3) E8 item 3.
    if (key == "coefficient_ring") {
        return {s.coefficient_ring.to_text(), s.coefficient_ring == CoefficientRingSetting(),
                s.coefficient_ring_source};
    }
    throw std::invalid_argument(
        "MathSettings::text: no canonical text for key \"" + key +
        "\" (unknown, or registered in keys() without an arm here)");
}

} // namespace

bool MathSettings::is_default() const {
    for (const std::string& key : registry()) {
        if (!view_of(*this, key).at_default) return false;
    }
    return true;
}

// In registry() order.  A key at its default contributes NOTHING, so the
// all-defaults text is empty and the fingerprint is 0.
std::string MathSettings::canonical_text() const {
    std::ostringstream oss;
    for (const std::string& key : registry()) {
        const KeyView v = view_of(*this, key);
        if (v.at_default) continue;
        oss << key << " = " << v.text << "\n";
    }
    return oss.str();
}

uint64_t MathSettings::fingerprint() const {
    const std::string text = canonical_text();
    if (text.empty()) return 0;
    return engine::FnvBytes(engine::kFnvOffset64, text);
}

std::string MathSettings::to_json() const {
    std::ostringstream oss;
    oss << "{";
    for (const std::string& key : registry()) {
        const KeyView v = view_of(*this, key);
        oss << "\"" << key << "\":{\"value\":\"" << v.text << "\",\"source\":\""
            << to_string(v.source) << "\"},";
    }
    oss << "\"fingerprint\":\"";
    // Fixed 16-hex-digit rendering, the HeckeReport convention.
    static const char* kHex = "0123456789abcdef";
    const uint64_t fp = fingerprint();
    for (int i = 15; i >= 0; --i) oss << kHex[(fp >> (4 * i)) & 0xfu];
    oss << "\"}";
    return oss.str();
}

// The canonical text of one key's value -- `set()`'s inverse, and the accessor
// the two writers above are built on, so the three agree byte for byte by
// construction and not by repetition.
std::string MathSettings::text(const std::string& key) const {
    return view_of(*this, key).text;
}

bool MathSettings::set(const std::string& key, const std::string& value, SettingSource src) {
    if (key == "locality") {
        locality = ParseLocality(value);
        locality_source = src;
        return true;
    }
    if (key == "u_ell") {
        u_ell = ParseUEll(value);
        u_ell_source = src;
        return true;
    }
    if (key == "sturm") {
        sturm = ParseSturm(value);
        sturm_source = src;
        return true;
    }
    if (key == "sturm_unavailable") {
        sturm_unavailable = ParseSturmUnavailable(value);
        sturm_unavailable_source = src;
        return true;
    }
    if (key == "commutant_budget") {
        commutant_budget = ParseCommutantBudget(value);
        commutant_budget_source = src;
        return true;
    }
    if (key == "final_mode") {
        final_mode = ParseFinalMode(value);
        final_mode_source = src;
        return true;
    }
    if (key == "field_polynomial") {
        field_polynomial = ParseFieldPolynomial(value);
        field_polynomial_source = src;
        return true;
    }
    if (key == "split_field_report") {
        split_field_report = ParseSplitFieldReport(value);
        split_field_report_source = src;
        return true;
    }
    if (key == "presentation") {
        presentation = ParsePresentation(value);
        presentation_source = src;
        return true;
    }
    if (key == "predictions") {
        predictions = ParsePredictions(value);
        predictions_source = src;
        return true;
    }
    if (key == "coefficient_ring") {
        coefficient_ring = ParseCoefficientRing(value);   // Z / Z/<m>: NotImplementedInThisBuild
        coefficient_ring_source = src;
        return true;
    }
    return false;
}

// Generic over keys(): set() stamps the source and text() is its inverse, so
// no per-key code (a key appended later is covered without an edit here).
MathSettings MathSettings::with_source(SettingSource src) const {
    MathSettings out = *this;
    for (const std::string& key : registry()) {
        const KeyView v = view_of(*this, key);
        out.set(key, v.text, v.at_default ? SettingSource::Default : src);
    }
    return out;
}

// ---------------------------------------------------------------------------
// The thread-local settings in force, and their propagation onto the task
// threads engine::tape::run_concurrently() starts (HeckeOperatorOnK.cpp:284
// is the pattern).
// ---------------------------------------------------------------------------
namespace {
thread_local MathSettings g_settings;

const bool g_registered = [] {
    engine::tape::register_thread_context([]() -> engine::tape::ThreadContextInstaller {
        const MathSettings snapshot = g_settings;
        return [snapshot]() -> engine::tape::ThreadContextGuard {
            g_settings = snapshot;
            return nullptr;
        };
    });
    return true;
}();
} // namespace

// (slot-8 WP-3) The refusal rule's guard -- see HeckeSettings.h.
uint32_t CoefficientRingK() { return CoefficientRingK("this construction site (a field or Z/ell^k pipeline object)"); }

uint32_t CoefficientRingK(const std::string& who) {
    const uint32_t k = math_settings().coefficient_ring.k;
    if (k == 0)
        RefuseCoefficientRing(who, 0, " (coefficient_ring=Z is honoured by the integral descent only: the descent record, "
                                      "the console verb descent, the campaign's descent cells; [C] thm:descentcert)");
    return k;
}

bool CoefficientRingIsZ() { return math_settings().coefficient_ring.k == 0; }

std::string CoefficientRingText(uint32_t k) {
    CoefficientRingSetting s;
    s.k = k;
    return s.to_text();
}

void RefuseCoefficientRing(const std::string& who, uint32_t k, const std::string& why) {
    // (slot-24 D6) Z is honoured in this build -- by the descent, not by `who`.
    throw NotImplementedInThisBuild(who + ": coefficient_ring=" + CoefficientRingText(k) +
                                    (k == 0 ? " is not honoured by this object" : " is not honoured in this build") + why);
}

void RequireFieldCoefficientRing(const std::string& who, uint32_t input_k) {
    const uint32_t k = CoefficientRingK(who);   // (slot-24 D6) Z: refused naming `who`
    if (k != 1) RefuseCoefficientRing(who, k);
    if (input_k != 1)
        RefuseCoefficientRing(who, input_k, " (an input object was constructed over it)");
}

const MathSettings& math_settings() {
    (void)g_registered;  // force the registration's static initialiser to be kept
    return g_settings;
}

void set_math_settings(const MathSettings& s) {
    (void)g_registered;
    g_settings = s;
}

} // namespace hecke
