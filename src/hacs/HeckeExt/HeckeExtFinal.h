// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

#pragma once
#ifndef HECKE_EXT_FINAL_H
#define HECKE_EXT_FINAL_H

// ===========================================================================
// hecke-extensions phase E2a (slot 4, WP-1) -- THE ONE TRI-STATE `Final`, its
// mode, and the FIXED reason list of [X] Sec. 1.3.1 R3.2.
//
// [X] = docs/implementation/hecke_extensions_implementation_plan.md v1.7;
// [E] = docs/latex/hecke_extensions.tex v1.4.2, Sec. `sec:schema` ("final is
// tri-state"); [C] = docs/latex/gl2_contract.tex v1.5.2, `sec:recordschema`.
//
// WHY A FILE OF ITS OWN.  [X] Sec. 1.3.1 R3.2 ends with "The tri-state is ONE
// type: HeckeGenerationCertificates reuses SturmFinal until E2's record layer
// owns `Final`, to which both units then migrate".  This is that migration:
//
//     hecke_ext::Final               the tri-state
//     hecke_ext::SturmFinal          = Final   (HeckeSturmBound.h, an alias)
//     hecke_ext::Finality            { value, reason }
//     hecke_ext::SturmFinality       = Finality (HeckeSturmBound.h, an alias)
//
// so `HeckeSturmBound`, `HeckeGenerationCertificates`, `HeckeSturmSweep` and
// `HeckeExtRecord` all speak about the SAME three values with the SAME reason
// vocabulary.  The JSON strings are UNCHANGED (`true` / `false` /
// `"heuristic"`), so no emitted text moves.
//
// DEPENDENCY-FREE, ON PURPOSE.  <cstdint> and <string> only.  HeckeSturmBound
// is the one unit of HeckeExt/ that consumes NOTHING of the pipeline
// (docs/matrix-engine/tools/check_heckeext_includes.sh rule 3), and it
// includes this header; the guard's rule-3 allow-list therefore names
// `HeckeExtFinal.h` with exactly that reason.  Everything here is `inline` or
// `constexpr`, so there is no TU and no link-time obligation either.
//
// THE REASON LIST IS FIXED ([X] R3.2, [E] Sec. 7).  A `final = false` carries
// a reason from
//
//     not_local, sturm_unavailable, sturm_heuristic, closure_unreachable,
//     certificate_failed:<code>, budget_exhausted:<kind>, interrupted
//
// and E2a adds the record-level state `not_applicable` for `n = 0` ([E]
// Def. 2.8: `dim M_k = 0`, so there is no claim to be final about).  The
// `<kind>` of `budget_exhausted` is one of R3.3's five instance kinds.  A
// free-form reason is a bug: `IsFinalReason()` is what a validation and a
// `verify()` check the emitted string against.
// ===========================================================================

#include <cstdint>
#include <string>

namespace hecke_ext {

// ---------------------------------------------------------------------------
// The tri-state ([X] Sec. 1.3.1 R3.2; [E] Sec. 7 "final is tri-state").
// ---------------------------------------------------------------------------
enum class Final : uint32_t { True = 0, False = 1, Heuristic = 2 };

// The JSON spelling, unchanged from the pre-E2a `SturmFinal`: `true` and
// `false` are JSON literals, `"heuristic"` a JSON string -- the caller adds
// the quotes for the third one only (every current caller emits the bare word
// into a text field, which is why this returns the bare word).
inline const char* ToString(Final value) {
    switch (value) {
        case Final::True:      return "true";
        case Final::False:     return "false";
        case Final::Heuristic: return "heuristic";
    }
    return "false";
}

// The value as a JSON VALUE: `true`, `false` or `"heuristic"` -- the form
// [E] Sec. 7's schema shows for the `final` key.
inline std::string ToJsonValue(Final value) {
    return value == Final::Heuristic ? std::string("\"heuristic\"")
                                     : std::string(ToString(value));
}

// ---------------------------------------------------------------------------
// `final_mode` ([X] Sec. 1.5 table, row 1; OQ8, slot-1 confirmation 8).
// ---------------------------------------------------------------------------
//
//   closure           the DEFAULT: an ideal is final only on a PROOF -- (a),
//                     (b) or (c) on its proof range, or (since hecke_ext/v4,
//                     slot-12 FIN) the finality certificate (d0) "dickson" /
//                     (d1) "boundary" of [E] sec:certgen.  `"heuristic"` is
//                     unreachable except through `sturm_unavailable =
//                     heuristic`.
//   sturm_heuristic   the user-selected mode of [E] Cert. `cert:sturm`'s
//                     HEURISTIC RANGE (Dickson-type m of X_4, and every m of
//                     X_2, X_3, X_5, X_6): a complete sweep off the proof
//                     range is `"heuristic"` instead of `false`.
//
// The enum lives here and not in Hecke/HeckeSettings.h for the same reason
// `Final` does: `is_final` is its only consumer and both belong to the record
// layer.  Hecke/HeckeSettings.h carries the SETTING (hecke::FinalMode) and
// parses it; the two are kept in step by HeckeExtRecordValidation.
enum class FinalMode : uint32_t { Closure = 0, SturmHeuristic = 1 };

inline const char* ToString(FinalMode mode) {
    return mode == FinalMode::SturmHeuristic ? "sturm_heuristic" : "closure";
}

// ---------------------------------------------------------------------------
// The fixed reason list.
// ---------------------------------------------------------------------------

// The seven bare reasons of [X] R3.2 plus E2a's record-level `not_applicable`.
inline constexpr const char* kReasonNotLocal          = "not_local";
inline constexpr const char* kReasonSturmUnavailable  = "sturm_unavailable";
inline constexpr const char* kReasonSturmHeuristic    = "sturm_heuristic";
inline constexpr const char* kReasonClosureUnreachable = "closure_unreachable";
inline constexpr const char* kReasonInterrupted       = "interrupted";
inline constexpr const char* kReasonNotApplicable     = "not_applicable";

// The two PREFIXED reasons.
inline constexpr const char* kReasonCertificateFailedPrefix = "certificate_failed:";
inline constexpr const char* kReasonBudgetExhaustedPrefix   = "budget_exhausted:";

// The five budget KINDS of [X] Sec. 1.3.1 R3.1 (the `resumable.kind` values),
// which are exactly the kinds `budget_exhausted:<kind>` may name.
inline constexpr const char* kKindSturmBound = "sturm_bound";
inline constexpr const char* kKindLocality   = "locality";
inline constexpr const char* kKindCommutant  = "commutant";
inline constexpr const char* kKindUEll       = "u_ell";
inline constexpr const char* kKindFill       = "fill";

inline bool IsResumableKind(const std::string& kind) {
    return kind == kKindSturmBound || kind == kKindLocality || kind == kKindCommutant ||
           kind == kKindUEll || kind == kKindFill;
}

// The certificate NAMES `certificate_failed:` may carry beside a clause code.
//
// WP-1 FINDING, for [C] `sec:recordschema` / [E] Sec. 7 / [X] Sec. 1.3.1 R3.2.
// All three write the reason as `certificate_failed:<code>`, "<code>" being
// the nonzero clause index a verify() returned.  The SHIPPED E1a/E1b code does
// not do that in one case: `generation_verdict` (HeckeGenerationCertificates,
// slot-2 WP-5) emits the literal `certificate_failed:sturm` when route (c)
// reports a swept T_q outside A_S -- a certificate NAME, because no verify()
// failed there; the sweep succeeded and its answer was "no".  That string is
// already emitted by the shipped code and pinned by
// HeckeExtSturmSweepValidation, so E2a ACCEPTS BOTH FORMS rather than moving
// an emitted value, and the documents should read `certificate_failed:<code
// or certificate name>` with this closed list.
//
// (slot-12 FIN, hecke_ext/v4; [E] v1.13.0 sec:certgen) "dickson" and
// "boundary" -- the finality certificates (d0) / (d1) -- join the list (the
// WP-12 proposal v2 Sec. 3.4 item 5).  The shipped driver lets a FAILED
// (d0) / (d1) fall through to the existing reasons (proposal Sec. 5 item 2),
// so it emits neither `certificate_failed:dickson` nor `:boundary`; the names
// are admitted so that a reader, a campaign tool or a later route may.
inline constexpr const char* kCertificateNames[] = {
    "cyclic", "commutant", "sturm", "locality", "module", "identity", "dickson", "boundary"
};

inline bool IsCertificateName(const std::string& name) {
    for (const char* c : kCertificateNames) {
        if (name == c) return true;
    }
    return false;
}

// `certificate_failed:<code>` -- <code> is the nonzero clause index a
// certificate's verify() returned.
inline std::string CertificateFailedReason(uint32_t code) {
    return std::string(kReasonCertificateFailedPrefix) + std::to_string(code);
}

// `certificate_failed:<certificate name>` -- the form the shipped
// `generation_verdict` emits for route (c)'s negative answer (see above).
inline std::string CertificateFailedReason(const std::string& certificate_name) {
    return std::string(kReasonCertificateFailedPrefix) + certificate_name;
}

// `budget_exhausted:<kind>` -- <kind> from the five above.
inline std::string BudgetExhaustedReason(const std::string& kind) {
    return std::string(kReasonBudgetExhaustedPrefix) + kind;
}

// True iff `reason` is a member of the fixed list -- the empty string
// included, which is what a `final = true` carries.  This is the predicate a
// record's verify() and HeckeExtRecordValidation run over every emitted
// reason, so an ad-hoc string cannot slip into the schema.
inline bool IsFinalReason(const std::string& reason) {
    if (reason.empty()) return true;
    if (reason == kReasonNotLocal || reason == kReasonSturmUnavailable ||
        reason == kReasonSturmHeuristic || reason == kReasonClosureUnreachable ||
        reason == kReasonInterrupted || reason == kReasonNotApplicable) {
        return true;
    }
    {
        const std::string p = kReasonCertificateFailedPrefix;
        if (reason.size() > p.size() && reason.compare(0, p.size(), p) == 0) {
            const std::string tail = reason.substr(p.size());
            if (tail.empty()) return false;
            if (IsCertificateName(tail)) return true;   // the shipped (c) form
            if (tail.size() > 10) return false;
            for (char c : tail) {
                if (c < '0' || c > '9') return false;
            }
            return true;
        }
    }
    {
        const std::string p = kReasonBudgetExhaustedPrefix;
        if (reason.size() > p.size() && reason.compare(0, p.size(), p) == 0) {
            return IsResumableKind(reason.substr(p.size()));
        }
    }
    return false;
}

// ---------------------------------------------------------------------------
// One unit's CONTRIBUTION to an ideal's `final`, or the ideal's own verdict.
// ---------------------------------------------------------------------------
//
// `reason` is "" exactly when `value == Final::True` ([X] R3.2: a `true`
// claims everything and explains nothing), and otherwise a member of the fixed
// list above.
struct Finality {
    Final       value = Final::True;
    std::string reason;

    bool operator==(const Finality& o) const {
        return value == o.value && reason == o.reason;
    }
    bool operator!=(const Finality& o) const { return !(*this == o); }

    // The structural invariant, checkable by any consumer.
    bool well_formed() const {
        if (value == Final::True) return reason.empty();
        return !reason.empty() && IsFinalReason(reason);
    }
};

} // namespace hecke_ext

#endif // HECKE_EXT_FINAL_H
