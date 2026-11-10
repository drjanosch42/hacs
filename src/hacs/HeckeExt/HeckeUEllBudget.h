// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

#pragma once
#ifndef HECKE_EXT_U_ELL_BUDGET_H
#define HECKE_EXT_U_ELL_BUDGET_H

// ===========================================================================
// (hecke-extensions slot 3, WP-1 / phase E1b-0) THE FROZEN U_ell RSS MODEL
// ([X] Sec. 1.5 `u_ell`, Sec. 2.2 Stage 1, [XA B3]; [OD] slot-1 confirmation 4)
// ===========================================================================
//
// Whether a run builds U_ell = T_p(ell) at p = ell is a MATHEMATICAL setting
// (`u_ell`, Hecke/HeckeSettings.h), not an engine knob: an omitted U_ell
// changes what the record contains.  [XA B3] makes the decision a pure
// function of (n, ell, budget_mb) through a FROZEN RSS model, so that the
// same cell decides the same way on every box and in every campaign split.
//
// THE MODEL.  U_ell's footprint is dominated by the Gamma_0(ell)\Gamma coset
// datum: m = ell + 1 cosets, each carrying rho_n(t_i) and rho_n(s_i^{-1}),
// two (n+1) x (n+1) matrices of uint32_t, plus O(1) exact 2x2 data.  Hence
//
//     rss_mb ~ (ell + 1) * (c0 + c1 * (n + 1)^2).
//
// This is the shape [XA B3] prescribes; WP-1 RE-FITTED its two constants.
//
// WHY TWO PARAMETERS AND NOT THREE.  The slot-3 plan allows a three-parameter
// model if the residuals exceed 15 %.  Two were fitted and REJECTED:
//   * (ell+1)(c0 + c1(n+1) + c2(n+1)^2)          rms 8.6 %, under-prediction -16.9 %
//   * (ell+1)(c0 + c1(n+1)^2) + c2(n+1)^2        rms 7.2 %, under-prediction -16.8 %
// Both improve the rms but NEITHER improves the worst under-prediction (the
// only direction a budget model can be wrong in dangerously), and both come
// out with a NEGATIVE third coefficient -- the second predicts a negative RSS
// at (n, ell) = (1000, 13), i.e. it would admit an arbitrarily large cell.
// A budget model must be monotone in n and in ell, so the two-parameter form
// stands and its residual band is documented instead.
//
// THE MEASUREMENT (docs/implementation/slot3/WP-1_report.md Sec. 4): 16 points
// of `--bench hecke n ell ell --repeat 1` on the reference box, over
// n in {16, 64, 128} x ell in {131, 691, 2003, 3617} plus (256, 131),
// (256, 691), (16, 5003) and (16, 10007).  Relative-error least squares.
// Residual band on the grid: +17.9 % / -15.9 %, rms 9.8 %.  A consumer that
// needs a hard guarantee must budget for the -16 % tail itself; the shipped
// `auto` mode does not, by design, because [XA B3]'s purpose is a
// REPRODUCIBLE verdict, not a memory guard.
//
// WHAT CHANGED against [X] Sec. 1.5's prose (which quoted the pre-measurement
// placeholder c0 = 0.11, c1 = 5.8e-5): c1 is ~3.9x larger, so the admitted
// n-ranges shrink -- `ell <= 13` to n <= 804 (not 1000), `ell = 691` to
// n <= 110 (not 220), `ell = 10007` to n <= 18 (not 38).  The VERDICTS on the
// ten Kummer cells of [E] Claim 1.8 are unchanged (seven admitted, and
// (16, 43867), (22, 2294797), (24, 657931) refused), which is what [X] Sec. 5
// relies on.  HeckeUEllBudgetValidation re-derives the whole table and pins
// the refusal at (16, 43867).
//
// PLACEMENT.  HeckeExt/, because the consumer is HeckeExtDriver's Stage 1
// (E1b, slot-3 WP-2) and nothing in the pipeline may depend on the extension
// layer (check_heckeext_includes.sh rule 1).  It is a PURE header: no
// pipeline object, no allocation, no I/O -- only <cstdint>/<string> and
// Hecke/HeckeSettings.h for the setting's own type.

#include <cstdint>
#include <string>

#include "../Hecke/HeckeSettings.h"

namespace hecke_ext {

// The frozen constants' version.  It enters the record beside the verdict
// ([X] Sec. 1.3.1 R3.3 item 4: "the frozen constants' version" is a decision
// input), so a re-fit is visible rather than silent.  BUMP IT on any change
// to kUEllRssC0 / kUEllRssC1.
inline constexpr const char* kUEllRssModelVersion = "u_ell_rss/v2-2026-09-17";

// rss_mb = (ell + 1) * (c0 + c1 * (n + 1)^2), MB.
inline constexpr double kUEllRssC0 = 0.123152;
inline constexpr double kUEllRssC1 = 2.24985e-4;

// The measured residual band of the frozen fit, for the record and for
// HeckeUEllBudgetValidation's own gate (a re-fit that leaves this band must
// bump the version).
inline constexpr double kUEllRssWorstOverPredict = 0.1788;   // +17.88 %, at (256, 131)
inline constexpr double kUEllRssWorstUnderPredict = 0.1595;  // -15.94 %, at (128, 3617)

// Predicted peak RSS in MB of `--bench hecke n ell ell` (i.e. of a run that
// builds U_ell), a pure function of (n, ell).  Monotone in both arguments.
inline double PredictUEllRssMb(uint32_t n, uint64_t ell) {
    const double np1 = static_cast<double>(n) + 1.0;
    return (static_cast<double>(ell) + 1.0) * (kUEllRssC0 + kUEllRssC1 * np1 * np1);
}

// The verdict's fixed reason vocabulary ([X] Sec. 1.5 / Sec. 7: the record
// key `u_ell.reason`).
enum class UEllReason : uint32_t { WithinBudget = 0, OverBudget = 1, Off = 2 };

inline const char* to_string(UEllReason r) {
    switch (r) {
        case UEllReason::WithinBudget: return "within_budget";
        case UEllReason::OverBudget:   return "u_ell_over_budget";
        case UEllReason::Off:          return "off";
    }
    return "u_ell_over_budget";
}

// The record object of [X] Sec. 2.2 Stage 1 (minus `resumable`, which is E2's:
// the resume is "re-run Stage 1 with U_ell attached to the same family", and
// that needs HeckeExtDriver's Stage 1-5 orchestration).
struct UEllVerdict {
    bool included = false;
    UEllReason reason = UEllReason::OverBudget;
    double predicted_mb = 0.0;
    uint32_t budget_mb = 0;
    const char* model_version = kUEllRssModelVersion;

    bool operator==(const UEllVerdict& o) const {
        return included == o.included && reason == o.reason &&
               predicted_mb == o.predicted_mb && budget_mb == o.budget_mb &&
               std::string(model_version) == std::string(o.model_version);
    }
};

// THE decision ([XA B3]: include U_ell iff the frozen model predicts <= M).
// Pure: no state, no settings lookup, no allocation beyond the returned POD.
// A budget of 0 is not reachable through the `u_ell` key (ParseUEll rejects
// budget_mb=0 and points at `off`), but is handled here as "admits nothing".
inline UEllVerdict UEllDecision(uint32_t n, uint64_t ell, uint32_t budget_mb) {
    UEllVerdict v;
    v.predicted_mb = PredictUEllRssMb(n, ell);
    v.budget_mb = budget_mb;
    v.included = (v.predicted_mb <= static_cast<double>(budget_mb));
    v.reason = v.included ? UEllReason::WithinBudget : UEllReason::OverBudget;
    return v;
}

// The same decision through the `u_ell` setting in force.  `off` short-circuits
// WITHOUT forming a prediction, so its verdict carries predicted_mb = 0 and
// reason = off -- the record then says "the user said no", never "it was too
// big".
inline UEllVerdict UEllDecision(uint32_t n, uint64_t ell, const hecke::UEllSetting& s) {
    if (s.mode == hecke::UEllMode::Off) {
        UEllVerdict v;
        v.included = false;
        v.reason = UEllReason::Off;
        v.predicted_mb = 0.0;
        v.budget_mb = s.effective_budget_mb();
        return v;
    }
    return UEllDecision(n, ell, s.effective_budget_mb());
}

// The largest even n this budget admits at `ell` (0 if none; even because odd
// n vanishes on this model -- the -I / odd-n gap).  Reporting helper for the
// admission table; PredictUEllRssMb is monotone in n, so a scan is exact.
inline uint32_t MaxAdmittedN(uint64_t ell, uint32_t budget_mb, uint32_t n_cap = 2000) {
    uint32_t best = 0;
    bool any = false;
    for (uint32_t n = 0; n <= n_cap; n += 2) {
        if (PredictUEllRssMb(n, ell) <= static_cast<double>(budget_mb)) {
            best = n;
            any = true;
        } else {
            break;
        }
    }
    return any ? best : 0u;
}

// True iff `ell` admits nothing at all at this budget (not even n = 0).
inline bool AdmitsNothing(uint64_t ell, uint32_t budget_mb) {
    return PredictUEllRssMb(0u, ell) > static_cast<double>(budget_mb);
}

} // namespace hecke_ext

#endif // HECKE_EXT_U_ELL_BUDGET_H
