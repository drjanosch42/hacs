// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// hecke-extensions slot 3, WP-1 (phase E1b-0) -- see
// HeckeUEllBudgetValidation.h for the arm list and HeckeUEllBudget.h for the
// model, its rejected three-parameter alternatives and its residual band.
//
// This TU obeys check_heckeext_includes.sh rule 2 (no Console/, no Cuda/, no
// Engine/, no BlockFlatMatrix.h) and rule 4 (no unordered_), which is why the
// `u_ell` PROPAGATION arm -- the only one needing
// engine::tape::run_concurrently -- lives in Hecke/HeckeUEllReachValidation.cpp
// instead.  Everything here is a closed form.

#include "HeckeUEllBudgetValidation.h"

#include "HeckeUEllBudget.h"
#include "../Hecke/HeckeSettings.h"
#include "../Util/ValidationCheck.h"

#include <cassert>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

namespace {

using hecke::MathSettings;
using hecke::SettingSource;
using hecke::UEllMode;
using hecke::UEllSetting;
using hecke_ext::PredictUEllRssMb;
using hecke_ext::UEllDecision;
using hecke_ext::UEllReason;
using hecke_ext::UEllVerdict;

oc_util::Checker Check("  *** FAIL [u_ell budget] ");

// --- (a) the frozen model against the measured grid ------------------------
//
// The 16 points of docs/implementation/slot3/WP-1_report.md Sec. 4, measured
// with `--bench hecke n ell ell --repeat 1` on the reference box.  They are
// pinned HERE so that a re-fit that changes the constants without re-measuring
// is caught: the model's residual band is part of the frozen model.
struct GridPoint {
    uint32_t n;
    uint64_t ell;
    double rss_mb;
};

const GridPoint kGrid[] = {
    {  16,   131,    28.1 }, {  64,   131,   135.3 }, { 128,   131,   454.1 },
    {  16,   691,   118.8 }, {  64,   691,   764.0 }, { 128,   691,  2708.5 },
    {  16,  2003,   360.4 }, {  64,  2003,  2413.8 }, { 128,  2003,  8752.8 },
    {  16,  3617,   668.7 }, {  64,  3617,  4592.2 }, { 128,  3617, 16645.3 },
    { 256,   131,  1677.8 }, { 256,   691,  9781.9 },
    {  16,  5003,   952.5 }, {  16, 10007,  1924.1 },
};

void ArmFitResiduals() {
    double worst_over = 0.0, worst_under = 0.0;
    for (const GridPoint& g : kGrid) {
        const double pred = PredictUEllRssMb(g.n, g.ell);
        const double rel = (pred - g.rss_mb) / g.rss_mb;
        if (rel > worst_over) worst_over = rel;
        if (-rel > worst_under) worst_under = -rel;
    }
    Check(worst_over <= hecke_ext::kUEllRssWorstOverPredict + 1e-6,
          "the frozen fit over-predicts by more than its declared band (" +
              std::to_string(worst_over * 100.0) + " %)");
    Check(worst_under <= hecke_ext::kUEllRssWorstUnderPredict + 1e-6,
          "the frozen fit under-predicts by more than its declared band (" +
              std::to_string(worst_under * 100.0) + " %)");
    // The band is DECLARED TIGHT: it must be attained, not padded, so that a
    // future re-fit cannot hide behind a generous band without bumping the
    // model version.
    Check(worst_over >= hecke_ext::kUEllRssWorstOverPredict - 0.01,
          "the declared over-prediction band is not attained (padded band)");
    Check(worst_under >= hecke_ext::kUEllRssWorstUnderPredict - 0.01,
          "the declared under-prediction band is not attained (padded band)");
    std::cout << "  [a] frozen fit " << hecke_ext::kUEllRssModelVersion << ": c0="
              << hecke_ext::kUEllRssC0 << " c1=" << hecke_ext::kUEllRssC1
              << ", residuals +" << worst_over * 100.0 << " % / -" << worst_under * 100.0
              << " % on " << (sizeof(kGrid) / sizeof(kGrid[0])) << " measured points"
              << std::endl;
}

// --- (b) monotonicity and purity -------------------------------------------

void ArmMonotonicityAndPurity() {
    for (uint64_t ell : { 2ull, 13ull, 131ull, 691ull, 3617ull, 10007ull, 43867ull }) {
        double prev = -1.0;
        for (uint32_t n = 0; n <= 512; n += 8) {
            const double p = PredictUEllRssMb(n, ell);
            Check(p > prev, "PredictUEllRssMb not strictly increasing in n at ell = " +
                                std::to_string(ell));
            prev = p;
        }
    }
    for (uint32_t n : { 0u, 16u, 128u }) {
        double prev = -1.0;
        for (uint64_t ell : { 2ull, 13ull, 131ull, 691ull, 2003ull, 3617ull, 10007ull,
                              43867ull, 657931ull, 2294797ull, 134217757ull }) {
            const double p = PredictUEllRssMb(n, ell);
            Check(p > prev, "PredictUEllRssMb not strictly increasing in ell at n = " +
                                std::to_string(n));
            prev = p;
        }
    }
    // Purity: the verdict depends on (n, ell, budget) alone. Take it under two
    // very different settings in force; nothing may change.
    const UEllVerdict a = UEllDecision(16u, 43867ull, 2048u);
    UEllVerdict b;
    {
        UEllSetting off;
        off.mode = UEllMode::Off;
        const hecke::ScopedUEll guard(off);
        b = UEllDecision(16u, 43867ull, 2048u);
    }
    Check(a == b, "UEllDecision(n, ell, M) is not pure (it saw the settings in force)");
    std::cout << "  [b] monotone in n and in ell on 7 x 65 + 3 x 11 points; the (n, ell, M) "
                 "overload is pure." << std::endl;
}

// --- (c) the admission table, re-derived -----------------------------------

void ArmAdmissionTable() {
    const uint32_t M = hecke::kDefaultUEllBudgetMb;

    // [X] Sec. 1.5's prose, written against the PRE-MEASUREMENT placeholder
    // (c0 = 0.11, c1 = 5.8e-5). Re-derived here and REPORTED, per the slot-3
    // plan: "the admission table will move; record the new table, do not treat
    // the change as a regression".
    struct Claim { uint64_t ell; uint32_t claimed_max_n; };
    const Claim claims[] = { { 13, 1000 }, { 691, 220 }, { 10007, 38 } };
    for (const Claim& c : claims) {
        const uint32_t got = hecke_ext::MaxAdmittedN(c.ell, M, 2000u);
        std::cout << "  [c] ell=" << c.ell << ": [X] Sec. 1.5 claimed n <= " << c.claimed_max_n
                  << ", the re-fitted model admits n <= " << got
                  << (got == c.claimed_max_n ? "  (unchanged)" : "  (CHANGED -- report to [X])")
                  << std::endl;
        // The claim is reported, not asserted. What IS asserted is internal
        // consistency: the reported maximum is admitted and the next even n is
        // not.
        Check(UEllDecision(got, c.ell, M).included,
              "MaxAdmittedN is not itself admitted at ell = " + std::to_string(c.ell));
        Check(!UEllDecision(got + 2u, c.ell, M).included,
              "MaxAdmittedN + 2 is admitted at ell = " + std::to_string(c.ell));
    }

    // The ten Kummer cells of [E] Claim 1.8 with their [X] Sec. 5 verdicts.
    // THESE the plan does rely on, so they are asserted.
    struct Kummer { uint32_t n; uint64_t ell; bool expect_included; };
    const Kummer kummer[] = {
        { 10,     691, true  }, { 14,    3617, true  }, { 16,   43867, false },
        { 18,     283, true  }, { 18,     617, true  }, { 20,     131, true  },
        { 20,     593, true  }, { 22,     103, true  }, { 22, 2294797, false },
        { 24,  657931, false },
    };
    for (const Kummer& k : kummer) {
        const UEllVerdict v = UEllDecision(k.n, k.ell, M);
        Check(v.included == k.expect_included,
              "Kummer cell (" + std::to_string(k.n) + ", " + std::to_string(k.ell) +
                  "): u_ell admission changed (predicted " + std::to_string(v.predicted_mb) +
                  " MB)");
        Check(v.reason == (k.expect_included ? UEllReason::WithinBudget : UEllReason::OverBudget),
              "Kummer cell (" + std::to_string(k.n) + ", " + std::to_string(k.ell) +
                  "): wrong reason");
    }
    // [XA B3]: no ell = 134217757 U_ell is ever built.
    Check(hecke_ext::AdmitsNothing(134217757ull, M),
          "ell = 134217757 admits some n at M = 2048 ([XA B3] requires none)");
    std::cout << "  [c] ten Kummer cells: 7 within_budget, 3 u_ell_over_budget; "
                 "ell = 134217757 admits nothing." << std::endl;
}

// --- (d) the pinned refusal -------------------------------------------------

void ArmPinnedRefusal() {
    const UEllVerdict v = UEllDecision(16u, 43867ull, 2048u);
    Check(!v.included, "UEllDecision(16, 43867, 2048).included is not false");
    Check(v.reason == UEllReason::OverBudget,
          "UEllDecision(16, 43867, 2048).reason is not u_ell_over_budget");
    Check(std::string(to_string(v.reason)) == "u_ell_over_budget",
          "the reason's spelling is not the record's");
    Check(v.budget_mb == 2048u, "UEllDecision(16, 43867, 2048).budget_mb is not 2048");
    Check(v.predicted_mb > 2048.0, "the pinned refusal's prediction is not above the budget");
    // The prediction is pinned to a band, not to a bit pattern: the point of
    // the pin is the VERDICT and the order of magnitude, and a re-fit that
    // bumps kUEllRssModelVersion is allowed to move the number inside it.
    Check(v.predicted_mb > 4000.0 && v.predicted_mb < 20000.0,
          "the pinned refusal's prediction (" + std::to_string(v.predicted_mb) +
              " MB) left its recorded band 4-20 GB");
    Check(std::string(v.model_version) == std::string(hecke_ext::kUEllRssModelVersion),
          "the verdict does not carry the frozen model's version");
    std::cout << "  [d] PIN UEllDecision(16, 43867, 2048) = {included=false, reason="
              << to_string(v.reason) << ", predicted_mb=" << v.predicted_mb
              << ", budget_mb=" << v.budget_mb << ", model=" << v.model_version << "}"
              << std::endl;
}

// --- (e) off, and a zero budget ---------------------------------------------

void ArmOffAndZero() {
    UEllSetting off;
    off.mode = UEllMode::Off;
    const UEllVerdict v = UEllDecision(10u, 691ull, off);
    Check(!v.included, "u_ell = off admitted a cell");
    Check(v.reason == UEllReason::Off, "u_ell = off did not report reason = off");
    Check(v.predicted_mb == 0.0,
          "u_ell = off formed a prediction (the record must say 'the user said no')");
    // (10, 691) IS within budget, so `off` is really short-circuiting.
    Check(UEllDecision(10u, 691ull, 2048u).included,
          "(10, 691) is not within budget -- the `off` arm would be vacuous");

    Check(!UEllDecision(0u, 2ull, 0u).included, "budget_mb = 0 admitted a cell");
    // And the `auto` overload agrees with the explicit one at the default.
    Check(UEllDecision(10u, 691ull, UEllSetting()) == UEllDecision(10u, 691ull, 2048u),
          "u_ell = auto does not decide like budget_mb = 2048");
    std::cout << "  [e] off short-circuits without a prediction; budget 0 admits nothing; "
                 "auto == budget_mb=2048." << std::endl;
}

// --- (f) the `u_ell` mathematical setting -----------------------------------

void ArmSetting() {
    // Defaults.
    {
        const MathSettings& d = MathSettings::defaults();
        Check(d.is_default(), "the default MathSettings is not default");
        Check(d.u_ell.mode == UEllMode::Auto &&
                  d.u_ell.budget_mb == hecke::kDefaultUEllBudgetMb,
              "u_ell does not default to auto / 2048 ([OD] slot-1 confirmation 4)");
        Check(d.canonical_text().empty() && d.fingerprint() == 0,
              "the all-defaults canonical text / fingerprint moved");
    }
    // Parsing, and the round trip through to_text().
    {
        Check(hecke::ParseUEll("auto") == UEllSetting(), "\"auto\" does not parse to the default");
        Check(hecke::ParseUEll("off").mode == UEllMode::Off, "\"off\" does not parse");
        const UEllSetting b = hecke::ParseUEll("budget_mb=512");
        Check(b.mode == UEllMode::BudgetMb && b.budget_mb == 512u,
              "\"budget_mb=512\" does not parse");
        for (const char* s : { "auto", "off", "budget_mb=512", "budget_mb=2048" }) {
            Check(hecke::ParseUEll(s).to_text() == std::string(s),
                  std::string("\"") + s + "\" does not round-trip through to_text()");
        }
        // An explicitly pinned budget is NOT the default, even at the default
        // value: a setting is recorded, never inferred ([X] Sec. 1.5 rule 2).
        Check(hecke::ParseUEll("budget_mb=2048") != UEllSetting(),
              "budget_mb=2048 collapsed onto the default (the pin would be unrecorded)");
        for (const char* bad : { "", "Auto", "on", "budget_mb=", "budget_mb=0", "budget_mb=-1",
                                 "budget_mb=12x", "budget=2048", "2048" }) {
            bool threw = false;
            try {
                hecke::ParseUEll(bad);
            } catch (const std::invalid_argument&) {
                threw = true;
            } catch (const std::exception&) {
                threw = false;
            }
            Check(threw, std::string("ParseUEll accepted the illegal value \"") + bad + "\"");
        }
    }
    // The registry, the canonical text and the fingerprint.
    {
        Check(MathSettings::known_key("u_ell"), "u_ell is not a registered key");
        const std::vector<std::string>& keys = MathSettings::keys();
        // Keys are APPENDED, never inserted (slot-3 WP-2 appended `sturm` and
        // `sturm_unavailable` after `u_ell`), so the PREFIX is what the
        // fingerprint depends on and what this arm pins.
        Check(keys.size() >= 2 && keys[0] == "locality" && keys[1] == "u_ell",
              "the registry order changed (the fingerprint is a function of it; keys are "
              "APPENDED, never inserted)");

        MathSettings s;
        Check(s.set("u_ell", "off", SettingSource::Spec), "set(u_ell, off) returned false");
        Check(s.u_ell.mode == UEllMode::Off && s.u_ell_source == SettingSource::Spec,
              "set(u_ell, off) did not record the value with its source");
        Check(!s.is_default(), "a non-default u_ell still reports is_default()");
        Check(s.canonical_text() == "u_ell = off\n",
              "the canonical text of a non-default u_ell is wrong: \"" + s.canonical_text() +
                  "\"");
        Check(s.fingerprint() != 0, "a non-default u_ell fingerprints to 0");

        MathSettings t;
        Check(t.set("u_ell", "budget_mb=512", SettingSource::Console), "set(budget_mb=512) failed");
        Check(t.fingerprint() != s.fingerprint(),
              "two distinct u_ell values share a fingerprint");
        Check(t.canonical_text() == "u_ell = budget_mb=512\n", "wrong canonical text for a pin");

        // Both keys non-default: registry order, one line each.
        MathSettings u = t;
        Check(u.set("locality", "budget=7", SettingSource::Spec), "set(locality) failed");
        Check(u.canonical_text() == "locality = budget=7\nu_ell = budget_mb=512\n",
              "the two-key canonical text is not in registry order: \"" + u.canonical_text() +
                  "\"");
        Check(u.fingerprint() != t.fingerprint() && u.fingerprint() != s.fingerprint(),
              "the two-key fingerprint collides with a one-key one");

        // set() leaves *this untouched for an unknown key and throws for a bad
        // value on a known one (the ONE validation point).
        MathSettings v0 = t;
        Check(!v0.set("u_elll", "off", SettingSource::Spec), "set() accepted an unknown key");
        Check(v0 == t, "a rejected unknown key still mutated the settings");
        bool threw = false;
        try {
            v0.set("u_ell", "budget_mb=0", SettingSource::Spec);
        } catch (const std::invalid_argument&) {
            threw = true;
        }
        Check(threw, "set(u_ell, budget_mb=0) did not throw");

        // to_json() carries the key with its source and the fingerprint.
        const std::string j = t.to_json();
        Check(j.find("\"u_ell\":{\"value\":\"budget_mb=512\",\"source\":\"console\"}") !=
                  std::string::npos,
              "to_json() does not carry u_ell with its source: " + j);
        Check(MathSettings::defaults().to_json().find(
                  "\"fingerprint\":\"0000000000000000\"") != std::string::npos,
              "the defaults' JSON fingerprint is not zero");
    }
    // The scoped override restores.
    {
        UEllSetting off;
        off.mode = UEllMode::Off;
        {
            const hecke::ScopedUEll guard(off);
            Check(hecke::math_settings().u_ell.mode == UEllMode::Off,
                  "ScopedUEll did not install the value");
            Check(hecke::math_settings().u_ell_source == SettingSource::Console,
                  "ScopedUEll did not record the source");
        }
        Check(hecke::math_settings().u_ell == UEllSetting(),
              "ScopedUEll did not restore the default");
        Check(hecke::math_settings().is_default(),
              "the thread-local settings in force are not the defaults after the scope");
    }
    std::cout << "  [f] the `u_ell` key: defaults, parsing, round trip, registry order, "
                 "canonical text, fingerprint, JSON, scope." << std::endl;
}

} // namespace

uint32_t HeckeUEllBudgetValidation() {
    Check.reset();
    std::cout << "================= HeckeUEllBudgetValidation (slot 3 WP-1, E1b-0) "
                 "=================\n";
    try {
        ArmFitResiduals();
        ArmMonotonicityAndPurity();
        ArmAdmissionTable();
        ArmPinnedRefusal();
        ArmOffAndZero();
        ArmSetting();
    } catch (const std::exception& e) {
        std::cout << "HeckeUEllBudgetValidation: TOP-LEVEL EXCEPTION: " << e.what() << std::endl;
        ++Check.fails;
    }
    std::cout << "U_ELL_BUDGET checks=" << Check.checks << " failures=" << Check.fails << std::endl;
    return Check.fails;
}
