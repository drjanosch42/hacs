// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// hecke-extensions phases E2a / E2 cont. (slot 4 WP-1; slot 5 WP-1) -- the
// `hecke_ext/v2` writer, the one `is_final`, and the record's structural
// verify().  See HeckeExtReport.h and HeckeExtRecord.h for the contract, the
// [E] / [X] / [C] citations and what the v1 -> v2 bump moves.
//
// INCLUDE DISCIPLINE ([X] Sec. 1.1; check_heckeext_includes.sh rules 1, 2, 4):
// its own header (hence the HeckeExt/ record model and Hecke/HeckeSettings.h),
// ModArithmetic's EngineCheck.h and Fnv.h, and C++ standard headers.  No
// Console/, no Cuda/, no Engine/, no BlockFlatMatrix.h, no `unordered_`.

#include "HeckeExtReport.h"
#include "HeckeFinalityRoutes.h"
#include "../BuildInfo.h"

#include "../../ModArithmetic/EngineCheck.h"
#include "../../ModArithmetic/Fnv.h"
#include "../Util/TextIo.h"

#include <algorithm>
#include <fstream>
#include <ostream>
#include <sstream>
#include <stdexcept>

namespace hecke_ext {

// ---------------------------------------------------------------------------
// The commit ([E] Sec. 7 `commit`, [X] Sec. 1.4).
// ---------------------------------------------------------------------------
const char* SourceCommit() {
    return oc::GitCommit();
}

std::vector<uint64_t> AnemicPool(uint64_t ell) {
    static const uint64_t kPrimesTo31[] = {2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31};
    std::vector<uint64_t> out;
    for (uint64_t q : kPrimesTo31) {
        if (q != ell) out.push_back(q);
    }
    return out;
}

namespace {

uint64_t Fnv(const std::string& s) { return engine::FnvBytes(engine::kFnvOffset64, s); }

// The writer emits only literals, numbers and ENUMERATED strings ([X]
// Sec. 1.4), so no escaper is needed -- but a caller-supplied group name or
// reason must still not be able to break the JSON.  This is the guard, not an
// escaper: a string with a quote, a backslash or a control character is a bug
// in the producer and is refused.
void CheckPlain(const std::string& s, const char* what) {
    for (char c : s) {
        ENGINE_CHECK_MSG(c != '"' && c != '\\' && static_cast<unsigned char>(c) >= 0x20,
                         "HeckeExtReport: a record string is not a plain enumerated token");
        (void)what;
    }
}

void AppendU32List(std::ostream& os, const std::vector<uint32_t>& v) {
    os << "[";
    for (size_t i = 0; i < v.size(); ++i) { if (i) os << ","; os << v[i]; }
    os << "]";
}

void AppendU64List(std::ostream& os, const std::vector<uint64_t>& v) {
    os << "[";
    for (size_t i = 0; i < v.size(); ++i) { if (i) os << ","; os << v[i]; }
    os << "]";
}

void AppendMatrix(std::ostream& os, const ExtMatrix& M) {
    os << "[";
    for (uint32_t r = 0; r < M.rows; ++r) {
        if (r) os << ",";
        os << "[";
        for (uint32_t c = 0; c < M.cols; ++c) { if (c) os << ","; os << M.data[r][c]; }
        os << "]";
    }
    os << "]";
}

// [C] `sec:recordschema` item 5: the four per-complex fields, DERIVED from the
// complex's own degree count and never a maximum over the record.
void AppendComplex(std::ostream& os, const ComplexRecord& c) {
    if (!c.available) {
        os << "{\"status\":\"unavailable\"}";
        return;
    }
    ENGINE_CHECK_MSG(c.num_degrees > 0,
                     "HeckeExtReport: an available complex must name its degree count "
                     "(q_max, truncated, q_max_source and degrees are derived from it)");
    ENGINE_CHECK_MSG(!c.q_max_configured || c.truncated,
                     "HeckeExtReport: q_max_source = \"set\" is admissible only where the "
                     "complex is truncated ([C] sec:recordschema item 5)");
    const uint32_t q_max = c.num_degrees - 1u;
    os << "{\"status\":\"computed\",\"q_max\":" << q_max
       << ",\"truncated\":" << (c.truncated ? "true" : "false")
       << ",\"q_max_source\":\"" << (c.q_max_configured ? "set" : "derived")
       << "\",\"degrees\":[";
    for (uint32_t q = 0; q <= q_max; ++q) { if (q) os << ","; os << q; }
    os << "]";

    // [C] `sec:recordschema` item 5, "a complex built from others": a derived
    // complex names the complexes it is derived from.
    if (!c.derived_from.empty()) {
        os << ",\"derived_from\":[";
        for (size_t i = 0; i < c.derived_from.size(); ++i) {
            CheckPlain(c.derived_from[i], "derived_from");
            if (i) os << ",";
            os << "\"" << c.derived_from[i] << "\"";
        }
        os << "]";
    }
    if (!c.model.empty()) {
        CheckPlain(c.model, "complex model");
        os << ",\"model\":\"" << c.model << "\"";
    }
    if (c.has_group_cohomology_flag) {
        os << ",\"is_group_cohomology\":" << (c.is_group_cohomology ? "true" : "false");
    }

    // [X] Sec. 6.2 E5's `les_maps`, one object per REPORTED spot of the cone's
    // long exact sequence ([C] `eq:coneles`).  A reading that is meaningless
    // at an unasserted spot is `null`, never a default `false`.
    if (!c.les_maps.empty()) {
        os << ",\"les_maps\":[";
        for (size_t i = 0; i < c.les_maps.size(); ++i) {
            const LesMapSpot& sp = c.les_maps[i];
            CheckPlain(sp.term, "les_maps.term");
            if (i) os << ",";
            os << "{\"term\":\"" << sp.term << "\",\"q\":" << sp.q
               << ",\"in_built\":" << (sp.in_built ? "true" : "false")
               << ",\"out_built\":" << (sp.out_built ? "true" : "false")
               << ",\"asserted\":" << (sp.asserted ? "true" : "false")
               << ",\"dim_middle\":" << sp.dim_middle
               << ",\"rank_in\":" << sp.rank_in
               << ",\"rank_out\":" << sp.rank_out
               << ",\"composite_zero\":";
            if (sp.in_built && sp.out_built) os << (sp.composite_zero ? "true" : "false");
            else os << "null";
            os << ",\"exact\":";
            if (sp.asserted) os << (sp.exact ? "true" : "false");
            else os << "null";
            os << ",\"coker_in_dim\":" << sp.coker_in_dim << "}";
        }
        os << "]";
    }

    // [C] `sec:recordschema` item 6(e): the two reserved cone flags, first
    // emitted here.  Recorded, never branched on; a `false` is conformance.
    if (c.has_cone_flags) {
        os << ",\"phi_flat_identity\":" << (c.phi_flat_identity ? "true" : "false")
           << ",\"residual_flat_zero\":" << (c.residual_flat_zero ? "true" : "false");
    }
    if (!c.action_record_schema.empty()) {
        CheckPlain(c.action_record_schema, "action_record_schema");
        os << ",\"action_record\":\"" << c.action_record_schema << "\"";
    }
    os << "}";
}

// R3.1's `resumable` object, with the inline / sidecar rule of item 6.
// `stem` empty => inline and flag; `seen_kinds` disambiguates a second state
// of the same kind (see HeckeExtReport.h).
void AppendResumable(std::ostream& os, const ResumableRecord& r, const std::string& stem,
                     const std::string& dir, std::vector<std::pair<std::string, uint32_t>>& seen) {
    CheckPlain(r.kind, "resumable.kind");
    ENGINE_CHECK_MSG(IsResumableKind(r.kind),
                     "HeckeExtReport: resumable.kind is not one of the five R3.1 kinds");
    os << "{\"kind\":\"" << r.kind << "\""
       << ",\"complete\":" << (r.complete ? "true" : "false")
       << ",\"budget\":" << (r.budget_json.empty() ? "{}" : r.budget_json)
       << ",\"reached\":" << (r.reached_json.empty() ? "{}" : r.reached_json)
       << ",\"budget_history\":" << (r.budget_history_json.empty() ? "[]"
                                                                   : r.budget_history_json);
    if (!r.state_json.empty()) {
        const bool oversize = r.state_json.size() > 65536u;
        if (!oversize) {
            os << ",\"state\":" << r.state_json;
        } else if (!stem.empty()) {
            // The FIRST oversize state of a kind keeps [X]'s literal name; the
            // i-th (i >= 1) gets the `.<i>` disambiguator (HeckeExtReport.h).
            uint32_t index = 0;
            bool found = false;
            for (auto& kv : seen) {
                if (kv.first == r.kind) { index = ++kv.second; found = true; break; }
            }
            if (!found) seen.emplace_back(r.kind, 0u);
            const std::string suffix =
                index == 0 ? std::string() : ("." + std::to_string(index));
            const std::string rel = stem + ".state." + r.kind + suffix + ".json";
            const std::string path = dir.empty() ? rel : (dir + "/" + rel);
            std::ofstream out(path.c_str(), std::ios::binary);
            ENGINE_CHECK_MSG(out.good(), "HeckeExtReport: cannot write the state sidecar");
            out << r.state_json;
            out.close();
            os << ",\"state_ref\":{\"path\":\"" << rel << "\",\"fnv64\":\""
               << oc_util::HexU64(Fnv(r.state_json), false) << "\",\"bytes\":" << r.state_json.size() << "}";
        } else {
            os << ",\"state_oversize\":true,\"state_fnv64\":\"" << oc_util::HexU64(Fnv(r.state_json), false)
               << "\",\"state\":" << r.state_json;
        }
    }
    os << "}";
}

} // namespace

// ---------------------------------------------------------------------------
// is_final ([X] Sec. 1.3.1 R3.2)
// ---------------------------------------------------------------------------

Finality is_final(const FinalInputs& in, const hecke::MathSettings& m) {
    Finality f;

    // [E] Def. 2.8: there is no claim to be final about.
    if (in.not_applicable) {
        f.value = Final::False;
        f.reason = kReasonNotApplicable;
        return f;
    }
    // R3.3 item 5: an interruption at a step boundary.
    if (in.interrupted) {
        f.value = Final::False;
        f.reason = kReasonInterrupted;
        return f;
    }
    // R3.3 item 2: budget mode never PROVES non-locality -- a genuinely
    // non-local block splits and never survives the loop, so what survives is
    // a block whose locality is unknown under the budget.
    if (!in.local) {
        f.value = Final::False;
        f.reason = in.locality_complete ? std::string(kReasonNotLocal)
                                        : BudgetExhaustedReason(kKindLocality);
        return f;
    }
    if (!in.certificates_ok) {
        f.value = Final::False;
        f.reason = CertificateFailedReason(in.first_failing_code);
        return f;
    }

    // The generation half is E1a/E1b's own map, called and not re-implemented,
    // so the two verdicts cannot drift apart.
    SturmRouteStatus c = SturmRouteStatus::Unavailable;
    switch (in.sturm) {
        case SturmSweepOutcome::Passed: c = SturmRouteStatus::Passed; break;
        case SturmSweepOutcome::Failed: c = SturmRouteStatus::Failed; break;
        default: c = SturmRouteStatus::Unavailable; break;
    }
    const GenerationVerdict v = generation_verdict(in.cyclic_passed, in.commutant, c);
    if (v.final_value == Final::True) {
        f.value = Final::True;
        f.reason.clear();
        return f;
    }
    // (slot-12 FIN, hecke_ext/v4; [E] sec:certgen cert:dickson / cert:boundary)
    // The finality certificates (d0) / (d1) certify the same predicate as
    // (a)-(c) (cl:gen(b): T_infty|_V = A_S), so a passed one is a PROOF and
    // `true` in every mode; a failed one falls through to the reasons below.
    if (in.dickson_passed || in.boundary_passed) {
        f.value = Final::True;
        f.reason.clear();
        return f;
    }

    // The two user-selected heuristic modes, the only routes to `"heuristic"`
    // ([X] R3.2; never in the default configuration).
    if (in.sturm == SturmSweepOutcome::Heuristic) {
        const bool off_proof_range = !in.sturm_proof_range;
        const bool no_bound        = in.sturm_proof_range && !in.sturm_has_B_prime;
        const bool heuristic =
            (off_proof_range && m.final_mode == hecke::FinalMode::SturmHeuristic) ||
            (no_bound && m.sturm_unavailable == hecke::SturmUnavailablePolicy::Heuristic);
        f.value  = heuristic ? Final::Heuristic : Final::False;
        f.reason = kReasonSturmHeuristic;
        return f;
    }

    f.value  = Final::False;
    f.reason = v.reason.empty() ? std::string(kReasonSturmUnavailable) : v.reason;
    return f;
}

// ---------------------------------------------------------------------------
// IdealRecord / HeckeExtRecord
// ---------------------------------------------------------------------------

std::vector<const ResumableRecord*> IdealRecord::all_resumables() const {
    std::vector<const ResumableRecord*> out;
    if (sturm.present && sturm.resumable.present) out.push_back(&sturm.resumable);
    // v2: `u_ell` is NOT an ideal's.  It is the cell's, and it is counted once
    // by HeckeExtRecord::all_resumables() ([OD] slot-4 close-out 5).
    for (const ResumableRecord& r : resumables) {
        if (r.present) out.push_back(&r);
    }
    return out;
}

std::vector<const ResumableRecord*> HeckeExtRecord::all_resumables() const {
    std::vector<const ResumableRecord*> out;
    if (u_ell.present && u_ell.resumable.present) out.push_back(&u_ell.resumable);
    return out;
}

uint32_t HeckeExtRecord::resumable_items() const {
    uint32_t n = 0;
    for (const ResumableRecord* r : all_resumables()) {
        if (!r->complete) ++n;
    }
    for (const IdealRecord& id : ideals) {
        for (const ResumableRecord* r : id.all_resumables()) {
            if (!r->complete) ++n;
        }
    }
    return n;
}

const char* HeckeExtRecord::verify_message(uint32_t code) {
    switch (code) {
        case 0:  return "ok";
        case 1:  return "the schema string is not hecke_ext/v5 (kExtSchema)";
        case 2:  return "weight != n_sym + 2, or ell < 2";
        case 3:  return "an available complex does not name its degree count";
        case 4:  return "q_max_source = \"set\" on a complex that is not truncated";
        case 5:  return "sign is not one of all / plus / minus";
        case 6:  return "a final verdict is not well formed (value / reason disagree)";
        case 7:  return "a final reason is not in the fixed list of R3.2";
        case 8:  return "a final verdict is not what is_final() returns on its own inputs";
        case 9:  return "a resumable item's kind is not one of the five R3.1 kinds";
        case 10: return "resumable_items() disagrees with the incomplete-item count";
        case 11: return "an ideal's index-aligned lists disagree in length";
        case 12: return "an embedded module structure does not verify";
        case 13: return "coefficient_field is present on a block that is not local";
        case 14: return "the certificate name is not one of the six enumerated values";
        case 15: return "iota_sign is not the constant of rmk:iotasign";
        case 16: return "an ideal's sign is not the record's, or is not all/plus/minus";
        case 17: return "an iota degree record is not well formed";
        case 18: return "an unavailable complex carries content";
        case 19: return "dim_H1_shriek is claimed without an available boundary complex";
        case 20: return "the record-level u_ell object is not well formed";
        case 21: return "a per-ideal groups entry is not well formed";
        case 22: return "q_max_source = \"set\" on a DERIVED complex (derived_from non-empty)";
        case 23: return "the iota exact-comparison object is missing or not well formed (obl:iota(ii)'s rider)";
        case 24: return "an ideal carries a presentation although presentation = off, or an empty one";
        case 25: return "a sturm proof_range / proof_range_source is not well formed "
                        "(closed list, null iff method_undefined, cell_local needs B' and "
                        "anemic_pool = AnemicPool(ell))";
        case 26: return "the s_sep object is not well formed (computed only at an in-process "
                        "atlas ell; ascending; added within primes and operators_computed)";
        case 27: return "an atlas label is not well formed (computed only on a local ideal at an "
                        "in-process atlas ell; key over anemic_pool, width r; closed type list)";
        case 29: return "the predictions are not well formed (kummer ascending with k = n + 2; "
                        "closed prediction names; confirmed index-aligned; unpredicted names an "
                        "ambient group of the entry; no refuted prediction under assert)";
        case 28: return "a canonical split is missing or not well formed (present iff local and "
                        "split_field_report = canonical; ok = false is a certificate failure)";
        case 30: return "the finality routes d0 / d1 are not well formed (closed value lists; "
                        "d0 on an H0 ideal and d1 on an H1 ideal of the unsplit record only; "
                        "passed iff the is_final input; not_applicable:u_ell iff ell is not in "
                        "operator_primes; boundary_reps = R(ell) iff d1 is passed or failed:<p>; "
                        "not_run:budget iff the (d1) stream is over its cap; "
                        "certificate dickson / boundary iff that route carried it)";
        case 31: return "certificate (d0) does not re-derive from the record's stored t_p "
                        "(Proposition D on the operator_primes, [C] prop:heckeH0(1))";
        case 32: return "certificate (d1) does not re-derive from the record's boundary_data "
                        "(present iff d1 = passed; r1 h x D and t_p D x D over boundary_reps; "
                        "rank(r1) = rank_r1 = D (H1); t_p = the stored t_p on S; t_p commuting "
                        "with and inside A_S (H2); [E] v1.14.0-pre.1 cert:boundary)";
        default: return "unknown code";
    }
}

uint32_t HeckeExtRecord::verify() const {
    if (schema != kExtSchema) return 1;
    if (ell < 2 || weight != n_sym + 2u) return 2;

    const ComplexRecord* cs[3] = {&bulk, &boundary, &pair};
    for (const ComplexRecord* c : cs) {
        if (c->available && c->num_degrees == 0) return 3;
        if (c->q_max_configured && !c->truncated) return 4;
        // [C] v1.7.0 `sec:recordschema` item 5: `derived_from` non-empty IS the
        // record's own derived flag (slot-5 WP-11's ruling, Option A). `"set"`
        // is legal on a source complex only, so a complex that names the
        // complexes it is derived from may not ALSO claim `q_max_source =
        // "set"` -- that is a record claiming a user chose a degree bound no
        // user may choose on a derived complex, and it is refused here, not
        // merely by the (unrelated) truncation check above.
        if (c->q_max_configured && !c->derived_from.empty()) return 22;
        // [C] `sec:recordschema` item 6(c): an unavailable complex is
        // `{"status": "unavailable"}` and NOTHING else, so it may not carry
        // the v2 content either -- a claim about a complex this record does
        // not have.
        if (!c->available &&
            (!c->les_maps.empty() || c->has_cone_flags || !c->derived_from.empty() ||
             !c->model.empty() || c->has_group_cohomology_flag ||
             !c->action_record_schema.empty())) {
            return 18;
        }
    }
    if (sign != "all" && sign != "plus" && sign != "minus") return 5;
    if (iota_sign != "eisenstein_in_H1_plus") return 15;
    if (has_dim_H1_shriek && !boundary.available) return 19;

    // The cell-level u_ell object ([OD] slot-4 close-out 5).
    if (u_ell.present) {
        if (u_ell.reason != "within_budget" && u_ell.reason != "u_ell_over_budget" &&
            u_ell.reason != "off") {
            return 20;
        }
        // A `resumable` says "this decision can be taken again with a bigger
        // budget"; an INCLUDED verdict has nothing to resume.
        if (u_ell.resumable.present && u_ell.included) return 20;
        if (u_ell.resumable.present && !IsResumableKind(u_ell.resumable.kind)) return 9;
    } else if (u_ell.resumable.present) {
        return 20;
    }

    // [C] `obl:iota`(ii)'s RIDER ([OD] 2026-09-22 Q3).  A record that reports
    // the involution at all must SAY whether the exact Mat22 comparison
    // happened; leaving a reader to assume it is exactly what the rider
    // forbids.  Deliberately placed BEFORE the per-degree loop and behind no
    // gate of its own, so the missing-object case is reached on every record
    // that has `iota`, however well formed the degree objects are.
    if (!iota.empty() && !iota_exact.present) return 23;
    if (iota_exact.present) {
        if (iota_exact.rows_with_exact > iota_exact.relation_rows) return 23;
        if (iota_exact.rows_compared.size() > iota_exact.rows_with_exact) return 23;
        // The exact half did not run, so no row can have been compared: a
        // record claiming otherwise misreports its own run (rule A6).
        if (!iota_exact.ran && !iota_exact.rows_compared.empty()) return 23;
        for (size_t i = 0; i < iota_exact.rows_compared.size(); ++i) {
            if (iota_exact.rows_compared[i] >= iota_exact.relation_rows) return 23;
            if (i != 0 && iota_exact.rows_compared[i] <= iota_exact.rows_compared[i - 1]) return 23;
        }
    } else if (iota_exact.ran || iota_exact.relation_rows != 0 ||
               !iota_exact.rows_compared.empty()) {
        // An absent object carries nothing ([C] item 6(c), as for a complex).
        return 23;
    }

    // The per-degree involution objects (E3).
    for (size_t i = 0; i < iota.size(); ++i) {
        const IotaDegreeRecord& ir = iota[i];
        if (ir.status != "split" && ir.status != "not_applicable") return 17;
        if (ir.status == "split" && ir.dim_plus + ir.dim_minus != ir.dim_H) return 17;
        if (ir.operators_commuting > ir.operators) return 17;
        if (ir.eisenstein_status != "defined" && ir.eisenstein_status != "undefined" &&
            ir.eisenstein_status != "not_applicable") {
            return 17;
        }
        if (ir.eisenstein_status == "defined" && ir.eisenstein_dim_socle_over_F != 1u) return 17;
        for (size_t j = 0; j < i; ++j) {
            if (iota[j].degree == ir.degree) return 17;
        }
    }

    uint32_t incomplete = 0;
    for (const ResumableRecord* r : all_resumables()) {
        if (!IsResumableKind(r->kind)) return 9;
        if (!r->complete) ++incomplete;
    }
    for (const IdealRecord& id : ideals) {
        if (id.sign != sign) return 16;
        for (const IdealGroupMatch& g : id.groups) {
            if (g.group.empty()) return 21;
            // The witness is what makes the match checkable; a claimed match
            // without one is the key [E] sec:schema asked for, unfilled.
            if (g.matched && g.witness_poly.empty()) return 21;
            if (g.matched && g.failure != 0) return 21;
            if (!g.matched && !g.own && g.failure == 0) return 21;
        }
        if (!id.final.well_formed()) return 6;
        if (!IsFinalReason(id.final.reason)) return 7;
        if (!(is_final(id.final_inputs, settings) == id.final)) return 8;
        for (const ResumableRecord* r : id.all_resumables()) {
            if (!IsResumableKind(r->kind)) return 9;
            if (!r->complete) ++incomplete;
        }
        if (id.operator_primes.size() != id.t_p.size()) return 11;
        if (id.tangent_primes.size() != id.tangent_active.size()) return 11;
        if (id.coefficient_field.present &&
            id.coefficient_field.primes.size() != id.coefficient_field.P.size()) {
            return 11;
        }
        if (id.coefficient_field.present && !id.local) return 13;
        if (id.certificate != "cyclic" && id.certificate != "commutant" &&
            id.certificate != "sturm(B')" && id.certificate != "unavailable" &&
            id.certificate != "dickson" && id.certificate != "boundary") {
            return 14;
        }
        // (slot-12 FIN, hecke_ext/v4) The finality routes (d0) / (d1).
        {
            const CertificateRoutes& cr = id.certificate_routes;
            if (!IsDicksonRouteValue(cr.d0) || !IsBoundaryRouteValue(cr.d1)) return 30;
            if (id.final_inputs.dickson_passed != (cr.d0 == kRoutePassed)) return 30;
            if (id.final_inputs.boundary_passed != (cr.d1 == kRoutePassed)) return 30;
            const std::string own = id.groups.empty() ? std::string() : id.groups[0].group;
            if (cr.d0 != kRouteNotRun && (own != "H0" || id.sign != "all")) return 30;
            if (cr.d1 != kRouteNotRun && (own != "H1" || id.sign != "all")) return 30;
            const bool ell_in_S = std::find(id.operator_primes.begin(), id.operator_primes.end(),
                                            static_cast<uint64_t>(ell)) != id.operator_primes.end();
            for (const std::string* v : {&cr.d0, &cr.d1}) {
                if (*v == kRouteNotRun) continue;
                if ((*v == kRouteNotApplicableUEll) == ell_in_S) return 30;
            }
            const bool d1_reps = cr.d1 == kRoutePassed ||
                                 cr.d1.compare(0, 7, kRouteFailedPrefix) == 0;
            if (d1_reps) {
                if (ell < 5 || id.boundary_reps != BoundaryRepresentatives(ell)) return 30;
                if (cr.d1 != kRoutePassed &&
                    std::find(id.boundary_reps.begin(), id.boundary_reps.end(),
                              std::stoull(cr.d1.substr(7))) == id.boundary_reps.end()) {
                    return 30;
                }
            } else if (!id.boundary_reps.empty()) {
                return 30;
            }
            // (audit G1) the cost cap re-derives from (n, ell, u_ell.budget_mb).
            if (cr.d1 == kRouteNotRunBudget || d1_reps) {
                if (!u_ell.present) return 30;
                const bool within = BoundaryStreamWithinBudget(n_sym, ell, u_ell.budget_mb);
                if ((cr.d1 == kRouteNotRunBudget) == within) return 30;
            }
            if ((id.certificate == "dickson") != (cr.d0 == kRoutePassed)) return 30;
            if ((id.certificate == "boundary") != (cr.d1 == kRoutePassed)) return 30;
            // (d0) re-derives from the record alone (proposal v2 Sec. 3.4 item 3).
            if (cr.d0 != kRouteNotRun && cr.d0 != kRouteNotApplicableUEll &&
                DicksonRoute(n_sym, ell, id.operator_primes, id.t_p) != cr.d0) {
                return 31;
            }
            // (slot-13 WP-7, hecke_ext/v5) (d1) re-derives from the record's
            // own data ([E] v1.14.0-pre.1 cert:boundary; FIN audit G2).
            if (id.boundary_data.present != (cr.d1 == kRoutePassed)) return 32;
            if (id.boundary_data.present &&
                BoundaryDataRederive(ell, id.operator_primes, id.t_p, id.boundary_reps,
                                     id.boundary_data.rank_r1, id.boundary_data.r1,
                                     id.boundary_data.t) != 0) {
                return 32;
            }
        }
        for (const ModuleRecordEntry& mr : id.modules) {
            if (mr.structure.verify() != 0) return 12;
        }
        // (slot-7 WP-2) Present only under `presentation = on` ([E] sec:schema).
        if (id.presentation_present &&
            (settings.presentation != hecke::PresentationMode::On ||
             id.presentation_fragment.empty())) {
            return 24;
        }
        if (!id.presentation_present && !id.presentation_fragment.empty()) return 24;
        // (slot-7 WP-3, stage 2 item 3; [E] sec:schema (a)) proof_range.
        if (id.sturm.present) {
            const std::string& src = id.sturm.proof_range_source;
            const bool undefined = (src == kProofRangeMethodUndefined);
            if (src != kProofRangeCellLocal && src != kProofRangeCallerInput && !undefined) {
                return 25;   // the retired "dickson_count" included
            }
            if (id.sturm.has_proof_range == undefined) return 25;       // null iff undefined
            if (undefined && id.sturm.has_B_prime) return 25;           // the one known case
            if (src == kProofRangeCellLocal &&
                (!id.sturm.has_B_prime || anemic_pool != AnemicPool(ell))) {
                return 25;
            }
            // The value is what the Sturm route RAN with; undefined leaves the
            // verdict to `sturm_unavailable` exactly as before (input true).
            if (id.sturm.has_proof_range &&
                id.final_inputs.sturm_proof_range != id.sturm.proof_range) {
                return 25;
            }
            if (undefined && !id.final_inputs.sturm_proof_range) return 25;
        }
    }
    if (!anemic_pool.empty() && anemic_pool != AnemicPool(ell)) return 25;
    // (slot-7 WP-3, stage 2 item 4) s_sep.
    if (s_sep.computed) {
        if (!IsInProcessAtlasEll(ell)) return 26;
        if (!std::is_sorted(s_sep.primes.begin(), s_sep.primes.end()) ||
            !std::is_sorted(s_sep.added.begin(), s_sep.added.end())) {
            return 26;
        }
        for (uint64_t q : s_sep.primes) {
            if (q == ell) return 26;
        }
        for (uint64_t q : s_sep.added) {
            if (!std::binary_search(s_sep.primes.begin(), s_sep.primes.end(), q)) return 26;
            if (std::find(operators_computed.begin(), operators_computed.end(), q) ==
                operators_computed.end()) {
                return 26;
            }
        }
    } else if (!s_sep.primes.empty() || !s_sep.added.empty()) {
        return 26;
    }
    // (slot-7 WP-3, stage 2 item 5) the atlas labels -- after the record-level
    // anemic_pool / s_sep clauses they depend on.
    for (const IdealRecord& id : ideals) {
        if (!id.label.computed) continue;
        const LabelRecord& lb = id.label;
        if (!id.local || !IsInProcessAtlasEll(ell) || !s_sep.computed) return 27;
        if (lb.r == 0 || lb.key.size() != anemic_pool.size() || anemic_pool.empty()) return 27;
        for (const auto& e : lb.key) {
            if (e.size() != lb.r) return 27;
        }
        if (lb.type != "eisenstein" && lb.type != "cuspidal") return 27;
        if ((lb.type == "eisenstein") != lb.has_chi) return 27;
        if (lb.has_chi && (lb.r != 1u || lb.chi_a >= lb.chi_b)) return 27;
    }
    // (slot-7 WP-3, stage 2 item 6) the canonical split.
    const bool canonical = (settings.split_field_report == hecke::SplitFieldReport::Canonical);
    for (const IdealRecord& id : ideals) {
        if (id.split_present) {
            if (!id.local || !canonical || id.split_fragment.empty() || !id.split_ok) return 28;
            if (id.split_r == 0) return 28;
        } else {
            if (!id.split_fragment.empty()) return 28;
            if (id.local && id.coefficient_field.present && canonical) return 28;
        }
    }
    // (slot-7 WP-3, stage 2 item 7) predictions.
    for (size_t i = 0; i < predictions.kummer.size(); ++i) {
        if (i && predictions.kummer[i].k <= predictions.kummer[i - 1].k) return 29;
    }
    if (!predictions.kummer.empty()) {
        bool has_weight = false;
        for (const auto& kv : predictions.kummer) has_weight = has_weight || kv.k == weight;
        if (!has_weight) return 29;
    }
    const bool assert_mode = (settings.predictions == hecke::PredictionsMode::Assert);
    for (const IdealRecord& id : ideals) {
        if (id.confirmed.size() != id.predicted.size()) return 29;
        for (size_t i = 0; i < id.predicted.size(); ++i) {
            const std::string& p = id.predicted[i];
            const std::string name = p.substr(0, p.find(':'));
            if (p.find(':') == std::string::npos || name != kPredictionEisensteinCuspidal) return 29;
            if (id.confirmed[i] < -1 || id.confirmed[i] > 1) return 29;
            // Under `assert` a refuted prediction fails the run: no record says so.
            if (assert_mode && id.confirmed[i] == 0) return 29;
        }
        for (const std::string& g : id.unpredicted) {
            bool known = false;
            for (const ModuleRecordEntry& m : id.modules) known = known || m.group == g;
            if (!known) return 29;
        }
    }
    if (incomplete != resumable_items()) return 10;
    return 0;
}

// ---------------------------------------------------------------------------
// The writer
// ---------------------------------------------------------------------------

namespace {

void AppendIdeal(std::ostream& os, const IdealRecord& id, const HeckeExtRecord& rec,
                 const ExtReportOptions& opts,
                 std::vector<std::pair<std::string, uint32_t>>& seen) {
    os << "{\"id\":" << id.id << ",\"dim\":" << id.dim;
    // v2 (E3): which side of the involution's split this entry belongs to.
    CheckPlain(id.sign, "ideal sign");
    os << ",\"sign\":\"" << id.sign << "\"";

    // residue_degree XOR residue_degree_lower_bound, never conflated (A-N4).
    if (id.residue_degree_certified) {
        os << ",\"residue_degree\":" << id.residue_degree;
    } else {
        os << ",\"residue_degree_lower_bound\":" << id.residue_degree;
    }
    os << ",\"residue_degree_certified\":" << (id.residue_degree_certified ? "true" : "false");
    os << ",\"local\":" << (id.local ? "true" : "false");

    os << ",\"residue_witness\":[";
    for (size_t i = 0; i < id.residue_witness.size(); ++i) {
        if (i) os << ",";
        os << "[" << id.residue_witness[i].first << "," << id.residue_witness[i].second << "]";
    }
    os << "]";

    // The A-N4 rule that an unproven claim is never emitted under an
    // unqualified key: pi_w and the coefficient field appear iff `local`.
    if (id.local) {
        os << ",\"witness_poly\":";
        AppendU32List(os, id.witness_poly);
    }
    if (id.coefficient_field.present) {
        ENGINE_CHECK_MSG(id.local,
                         "HeckeExtReport: coefficient_field on a block that is not local");
        os << ",\"coefficient_field\":{\"w_hat_exponent\":"
           << id.coefficient_field.w_hat_exponent << ",\"P\":{";
        for (size_t i = 0; i < id.coefficient_field.primes.size(); ++i) {
            if (i) os << ",";
            os << "\"" << id.coefficient_field.primes[i] << "\":";
            AppendU32List(os, id.coefficient_field.P[i]);
        }
        os << "}}";
    }

    // (slot-7 WP-3, stage 2 item 5) The atlas label, always present
    // ([E] sec:schema (b)).
    os << ",\"label\":";
    if (!id.local) {
        os << "null";
    } else if (!id.label.computed) {
        os << "{\"status\":\"atlas_unavailable\"}";
    } else {
        const LabelRecord& lb = id.label;
        CheckPlain(lb.type, "label.type");
        os << "{\"status\":\"computed\",\"key\":[";
        for (size_t i = 0; i < lb.key.size(); ++i) {
            if (i) os << ",";
            AppendU32List(os, lb.key[i]);
        }
        os << "],\"r\":" << lb.r << ",\"system\":" << lb.system << ",\"type\":\"" << lb.type
           << "\",\"chi\":";
        if (lb.has_chi) os << "[" << lb.chi_a << "," << lb.chi_b << "]";
        else os << "null";
        os << ",\"base\":" << lb.base << ",\"theta_twist\":" << lb.theta_twist << "}";
    }

    // (slot-7 WP-3, stage 2 item 6) The canonical split, verbatim
    // ([E] sec:schema (c)); present iff local and split_field_report = canonical.
    if (id.split_present) {
        os << ",\"split\":" << id.split_fragment;
    }

    if (id.has_phase_reached) {
        CheckPlain(id.phase_reached, "phase_reached");
        os << ",\"phase_reached\":\"" << id.phase_reached << "\"";
    }

    if (id.algebra_present) {
        os << ",\"algebra\":{\"dim\":" << id.algebra_dim << ",\"embdim\":" << id.embdim
           << ",\"nilpotency\":" << id.nilpotency
           << ",\"gorenstein\":" << (id.gorenstein ? "true" : "false") << ",\"hilbert\":";
        AppendU32List(os, id.hilbert);
        os << "}";
    }

    CheckPlain(id.certificate, "certificate");
    CheckPlain(id.certificate_routes.a, "certificate_routes.a");
    CheckPlain(id.certificate_routes.b, "certificate_routes.b");
    CheckPlain(id.certificate_routes.c, "certificate_routes.c");
    CheckPlain(id.certificate_routes.d0, "certificate_routes.d0");
    CheckPlain(id.certificate_routes.d1, "certificate_routes.d1");
    os << ",\"certificate\":\"" << id.certificate << "\"";
    os << ",\"certificate_routes\":{\"a\":\"" << id.certificate_routes.a << "\",\"b\":\""
       << id.certificate_routes.b << "\",\"c\":\"" << id.certificate_routes.c
       << "\",\"d0\":\"" << id.certificate_routes.d0 << "\",\"d1\":\""
       << id.certificate_routes.d1 << "\"}";
    os << ",\"generators_certified\":";
    AppendU64List(os, id.generators_certified);
    // (slot-12 FIN, hecke_ext/v4) the classes (d1) tested -- not generators.
    os << ",\"boundary_reps\":";
    AppendU64List(os, id.boundary_reps);
    // (slot-13 WP-7, hecke_ext/v5) (d1)'s data, present iff d1 = passed.
    if (id.boundary_data.present) {
        ENGINE_CHECK_MSG(id.boundary_data.t.size() == id.boundary_reps.size(),
                         "HeckeExtReport: boundary_data.t is not index-aligned with boundary_reps");
        os << ",\"boundary_data\":{\"rank_r1\":" << id.boundary_data.rank_r1 << ",\"r1\":";
        AppendMatrix(os, id.boundary_data.r1);
        os << ",\"t\":{";
        for (size_t i = 0; i < id.boundary_data.t.size(); ++i) {
            if (i) os << ",";
            os << "\"" << id.boundary_reps[i] << "\":";
            AppendMatrix(os, id.boundary_data.t[i]);
        }
        os << "}}";
    }

    // The `sturm` object is EXACTLY [E] Sec. 7's four keys; its siblings carry
    // the rest, and `proof_range` is recorded as an INPUT.
    if (id.sturm.present) {
        CheckPlain(id.sturm.mode, "sturm.mode");
        CheckPlain(id.sturm.outcome, "sturm.outcome");
        CheckPlain(id.sturm.proof_range_source, "sturm.proof_range_source");
        os << ",\"sturm\":{\"B_prime\":";
        if (id.sturm.has_B_prime) os << id.sturm.B_prime; else os << "null";
        os << ",\"cap\":" << id.sturm.cap << ",\"closure_dim\":" << id.sturm.closure_dim
           << ",\"dim_M_k\":" << id.sturm.dim_M_k << ",\"mode\":\"" << id.sturm.mode
           << "\",\"outcome\":\"" << id.sturm.outcome << "\",\"sturm_primes\":";
        AppendU64List(os, id.sturm.sturm_primes);
        os << ",\"proof_range\":"
           << (!id.sturm.has_proof_range ? "null" : (id.sturm.proof_range ? "true" : "false"))
           << ",\"proof_range_source\":\"" << id.sturm.proof_range_source << "\"";
        if (id.sturm.resumable.present) {
            os << ",\"resumable\":";
            AppendResumable(os, id.sturm.resumable, opts.sidecar_stem, opts.sidecar_dir, seen);
        }
        os << "}";
    }

    // v2: `u_ell` is NOT here.  The admission of U_ell is a property of
    // (n, ell) and of the budget, so it is the RECORD's object and is emitted
    // once ([OD] slot-4 close-out 5; [E] sec:schema v1.5.0's booked repair).

    os << ",\"tangent\":{";
    for (size_t i = 0; i < id.tangent_primes.size(); ++i) {
        if (i) os << ",";
        os << "\"" << id.tangent_primes[i] << "\":\""
           << (id.tangent_active[i] ? "active" : "inactive") << "\"";
    }
    os << "}";

    os << ",\"t_p\":{";
    for (size_t i = 0; i < id.operator_primes.size(); ++i) {
        if (i) os << ",";
        os << "\"" << id.operator_primes[i] << "\":";
        AppendMatrix(os, id.t_p[i]);
    }
    os << "}";

    os << ",\"modules\":{";
    for (size_t i = 0; i < id.modules.size(); ++i) {
        CheckPlain(id.modules[i].group, "module group");
        if (i) os << ",";
        os << "\"" << id.modules[i].group << "\":" << id.modules[i].structure.to_json();
    }
    os << "}";

    // v2: Stage 2's cross-block matching, which v1 had no key for although
    // [XA M5] computes it ([E] sec:schema v1.5.0's second booked repair).
    // `modules` keeps its "one entry per ambient group" shape.
    os << ",\"groups\":[";
    for (size_t i = 0; i < id.groups.size(); ++i) {
        const IdealGroupMatch& g = id.groups[i];
        CheckPlain(g.group, "ideal groups entry");
        if (i) os << ",";
        os << "{\"group\":\"" << g.group << "\",\"own\":" << (g.own ? "true" : "false")
           << ",\"matched\":" << (g.matched ? "true" : "false")
           << ",\"ideal_id\":" << g.ideal_id
           << ",\"witness_poly\":";
        AppendU32List(os, g.witness_poly);
        os << ",\"w_hat_exponent\":" << g.w_hat_exponent
           << ",\"w_hat_exponent_other\":" << g.w_hat_exponent_other
           << ",\"failure\":" << g.failure << "}";
    }
    os << "]";

    // [E] `alg:stage5`(4): the canonical submodules of D6, keyed by the
    // ambient group.  v1 emitted the ideal-level placeholder
    // {"status": "unavailable"} because the four slots need E3 / E4 / E5
    // objects; v2 emits what it has and keeps the placeholder where it has
    // nothing ([C] `sec:recordschema` item 6(c)).
    os << ",\"canonical\":";
    if (id.canonical.empty()) {
        os << "{\"status\":\"unavailable\"}";
    } else {
        os << "{";
        for (size_t i = 0; i < id.canonical.size(); ++i) {
            const CanonicalPosition& cp = id.canonical[i];
            CheckPlain(cp.group, "canonical group");
            CheckPlain(cp.eisenstein, "canonical.eisenstein");
            CheckPlain(cp.H1_shriek, "canonical.H1_shriek");
            CheckPlain(cp.dickson, "canonical.dickson");
            CheckPlain(cp.L_image, "canonical.L_image");
            if (i) os << ",";
            os << "\"" << cp.group << "\":{\"eisenstein\":\"" << cp.eisenstein
               << "\",\"H1_shriek\":\"" << cp.H1_shriek
               << "\",\"dickson\":\"" << cp.dickson
               << "\",\"L_image\":\"" << cp.L_image << "\"}";
        }
        os << "}";
    }

    for (const ResumableRecord& r : id.resumables) {
        if (!r.present) continue;
        CheckPlain(r.kind, "resumable.kind");
        os << ",\"resumable_" << r.kind << "\":";
        AppendResumable(os, r, opts.sidecar_stem, opts.sidecar_dir, seen);
    }

    ENGINE_CHECK_MSG(id.final.well_formed(),
                     "HeckeExtReport: a final verdict is not well formed");
    ENGINE_CHECK_MSG(is_final(id.final_inputs, rec.settings) == id.final,
                     "HeckeExtReport: the stored final verdict is not is_final()'s");
    // (slot-7 WP-3, stage 2 item 7) always present ([E] sec:schema (d)).
    os << ",\"predicted\":[";
    for (size_t i = 0; i < id.predicted.size(); ++i) {
        CheckPlain(id.predicted[i], "predicted");
        os << (i ? "," : "") << "\"" << id.predicted[i] << "\"";
    }
    os << "],\"confirmed\":[";
    for (size_t i = 0; i < id.confirmed.size(); ++i) {
        os << (i ? "," : "")
           << (id.confirmed[i] > 0 ? "true" : (id.confirmed[i] == 0 ? "false" : "null"));
    }
    os << "],\"unpredicted\":[";
    for (size_t i = 0; i < id.unpredicted.size(); ++i) {
        CheckPlain(id.unpredicted[i], "unpredicted");
        os << (i ? "," : "") << "\"" << id.unpredicted[i] << "\"";
    }
    os << "]";
    os << ",\"final\":" << ToJsonValue(id.final.value) << ",\"final_reason\":";
    if (id.final.reason.empty()) os << "null";
    else os << "\"" << id.final.reason << "\"";
    // (slot-7 WP-2) Present ONLY under `presentation = on` ([E] sec:schema,
    // slot-7 owner confirmation 15): absent from the default record, so the
    // default text is byte-for-byte what it was before the key existed.
    if (id.presentation_present) {
        ENGINE_CHECK_MSG(rec.settings.presentation == hecke::PresentationMode::On &&
                             !id.presentation_fragment.empty(),
                         "HeckeExtReport: a presentation object without presentation = on");
        os << ",\"presentation\":{\"generator_basis\":\"block\",\"relation_basis\":"
              "\"algebra_rref\",\"fragment\":"
           << id.presentation_fragment << "}";
    }
    os << "}";
}

} // namespace

void StreamHeckeExtReportJson(std::ostream& os, const HeckeExtRecord& record,
                              const ExtReportOptions& opts) {
    ENGINE_CHECK_MSG(record.schema == kExtSchema,
                     "HeckeExtReport: the record's schema is not hecke_ext/v5 (kExtSchema)");
    ENGINE_CHECK_MSG(record.ell >= 2, "HeckeExtReport: ell must be a prime >= 2");
    ENGINE_CHECK_MSG(record.weight == record.n_sym + 2u,
                     "HeckeExtReport: weight must be n_sym + 2");
    ENGINE_CHECK_MSG(record.sign == "all" || record.sign == "plus" || record.sign == "minus",
                     "HeckeExtReport: sign must be all, plus or minus");
    CheckPlain(record.model, "model");
    CheckPlain(record.iota_sign, "iota_sign");
    CheckPlain(record.coefficient_ring, "coefficient_ring");

    std::vector<std::pair<std::string, uint32_t>> seen;

    os << "{\"schema\":\"" << kExtSchema << "\"";
    os << ",\"commit\":\"" << (record.commit.empty() ? SourceCommit() : record.commit.c_str())
       << "\"";
    os << ",\"model\":\"" << record.model << "\"";
    os << ",\"iota_sign\":\"" << record.iota_sign << "\"";
    os << ",\"n_sym\":" << record.n_sym << ",\"ell\":" << record.ell
       << ",\"weight\":" << record.weight << ",\"sign\":\"" << record.sign << "\"";
    os << ",\"coefficient_ring\":\"" << record.coefficient_ring << "\"";
    // In band, as hecke_report states them.
    os << ",\"poly_coeff_order\":\"low->high degree (A-N4)\"";
    os << ",\"matrix_entry_order\":\"arrays of rows (row-major); entries are canonical "
          "residues in [0, ell)\"";
    // [X] Sec. 1.5 rule 2: a setting is RECORDED, never inferred.
    os << ",\"settings\":" << record.settings.to_json();
    // (slot-7 WP-3, stage 2 item 7; [E] sec:schema (d)) beside `settings`,
    // because golden sees `settings` only as its fingerprint.
    os << ",\"predictions_mode\":\"" << hecke::to_text(record.settings.predictions) << "\"";
    os << ",\"predictions\":{\"kummer\":[";
    for (size_t i = 0; i < record.predictions.kummer.size(); ++i) {
        const PredictionsRecord::Kummer& kv = record.predictions.kummer[i];
        CheckPlain(kv.reason, "predictions.kummer.reason");
        if (i) os << ",";
        os << "{\"k\":" << kv.k << ",\"applicable\":" << (kv.applicable ? "true" : "false")
           << ",\"divides\":" << (kv.divides ? "true" : "false") << ",\"k_reduced\":";
        if (kv.has_reduced) os << kv.k_reduced; else os << "null";
        os << ",\"b_over_k_mod_ell\":";
        if (kv.has_reduced) os << kv.b_over_k_mod_ell; else os << "null";
        os << ",\"reason\":\"" << kv.reason << "\"}";
    }
    os << "],\"glover_factors\":"
       << (record.predictions.glover_json.empty() ? std::string("null")
                                                  : record.predictions.glover_json)
       << "}";
    os << ",\"resumable_items\":" << record.resumable_items();

    // v2: the CELL's U_ell verdict, emitted once, with ONE `resumable`.
    os << ",\"u_ell\":";
    if (!record.u_ell.present) {
        os << "{\"status\":\"unavailable\"}";
    } else {
        CheckPlain(record.u_ell.reason, "u_ell.reason");
        CheckPlain(record.u_ell.model_version, "u_ell.model_version");
        std::ostringstream mb;
        mb.setf(std::ios::fixed);
        mb.precision(2);
        mb << record.u_ell.predicted_mb;
        os << "{\"included\":" << (record.u_ell.included ? "true" : "false")
           << ",\"reason\":\"" << record.u_ell.reason << "\",\"predicted_mb\":" << mb.str()
           << ",\"budget_mb\":" << record.u_ell.budget_mb << ",\"model_version\":\""
           << record.u_ell.model_version << "\"";
        if (record.u_ell.resumable.present) {
            ENGINE_CHECK_MSG(!record.u_ell.included,
                             "HeckeExtReport: an included U_ell verdict has nothing to resume");
            os << ",\"resumable\":";
            AppendResumable(os, record.u_ell.resumable, opts.sidecar_stem, opts.sidecar_dir, seen);
        }
        os << "}";
    }

    if (record.convention_present) {
        os << ",\"convention_check\":{\"tau_mod_ell_ok\":"
           << (record.tau_mod_ell_ok ? "true" : "false") << ",\"eis_eigenvalue_ok\":"
           << (record.eis_eigenvalue_ok ? "true" : "false") << "}";
    }

    os << ",\"complexes\":{\"bulk\":";
    AppendComplex(os, record.bulk);
    os << ",\"boundary\":";
    AppendComplex(os, record.boundary);
    os << ",\"pair\":";
    AppendComplex(os, record.pair);
    os << "}";

    os << ",\"groups\":{";
    for (size_t i = 0; i < record.groups.size(); ++i) {
        CheckPlain(record.groups[i].name, "group name");
        if (i) os << ",";
        os << "\"" << record.groups[i].name << "\":";
        if (record.groups[i].available) os << "{\"dim\":" << record.groups[i].dim << "}";
        else os << "{\"status\":\"unavailable\"}";
    }
    if (!record.groups.empty()) os << ",";
    // [E] sec:schema v1.5.0: the CLOSED LIST of four integer scalars, which is
    // how a consumer tells a scalar from a group.  The fourth is emitted as
    // `null` where the boundary complex that defines it is unavailable, so
    // that the list is a constant of the schema.
    os << "\"dim_M_k\":" << record.dim_M_k << ",\"dim_S_k\":" << record.dim_S_k
       << ",\"dim_dickson\":" << record.dim_dickson << ",\"dim_H1_shriek\":";
    if (record.has_dim_H1_shriek) os << record.dim_H1_shriek; else os << "null";
    os << "}";

    os << ",\"les\":";
    if (!record.les_available) {
        os << "{\"status\":\"unavailable\"}";
    } else {
        os << "{\"exact\":" << (record.les_exact ? "true" : "false")
           << ",\"hecke_equivariant\":" << (record.les_hecke_equivariant ? "true" : "false")
           << ",\"ranks\":{";
        for (size_t i = 0; i < record.les_ranks.size(); ++i) {
            CheckPlain(record.les_ranks[i].first, "les rank name");
            if (i) os << ",";
            os << "\"" << record.les_ranks[i].first << "\":" << record.les_ranks[i].second;
        }
        os << "}}";
    }

    // v2 (E3): the computed involution objects beside the constant
    // `iota_sign`, one per degree it was computed in.
    os << ",\"iota\":";
    if (record.iota.empty()) {
        os << "{\"status\":\"unavailable\"}";
    } else {
        os << "[";
        for (size_t i = 0; i < record.iota.size(); ++i) {
            const IotaDegreeRecord& ir = record.iota[i];
            CheckPlain(ir.status, "iota.status");
            CheckPlain(ir.eisenstein_status, "iota.eisenstein.status");
            if (i) os << ",";
            os << "{\"degree\":" << ir.degree << ",\"status\":\"" << ir.status
               << "\",\"dim_H\":" << ir.dim_H;
            if (ir.status == "split") {
                os << ",\"dim_plus\":" << ir.dim_plus
                   << ",\"dim_minus\":" << ir.dim_minus;
            } else {
                os << ",\"dim_plus\":null,\"dim_minus\":null";
            }
            os << ",\"rank_iota\":" << ir.rank_iota << ",\"charpoly_iota\":";
            AppendU32List(os, ir.charpoly_iota);
            os << ",\"commutation_asserted\":"
               << (ir.commutation_asserted ? "true" : "false")
               << ",\"operators\":" << ir.operators
               << ",\"operators_commuting\":" << ir.operators_commuting
               << ",\"eisenstein\":{\"status\":\"" << ir.eisenstein_status << "\"";
            if (ir.eisenstein_applicable) {
                os << ",\"block\":" << ir.eisenstein_block << ",\"f\":" << ir.eisenstein_f
                   << ",\"dim_socle\":" << ir.eisenstein_dim_socle
                   << ",\"dim_socle_over_F\":" << ir.eisenstein_dim_socle_over_F
                   << ",\"socle_equals_mV\":"
                   << (ir.eisenstein_socle_equals_mV ? "true" : "false");
            }
            os << "}}";
        }
        os << "]";
    }

    // v2 ([OD] 2026-09-22 Q3): [C] `obl:iota`(ii)'s rider -- the record's own
    // statement of whether the exact Mat22 comparison happened, and on which
    // rows.  Recorded, never branched on (rule A6).
    os << ",\"iota_exact_comparison\":";
    if (!record.iota_exact.present) {
        os << "{\"status\":\"unavailable\"}";
    } else {
        os << "{\"ran\":" << (record.iota_exact.ran ? "true" : "false")
           << ",\"relation_rows\":" << record.iota_exact.relation_rows
           << ",\"rows_with_exact\":" << record.iota_exact.rows_with_exact
           << ",\"rows_compared\":";
        AppendU32List(os, record.iota_exact.rows_compared);
        os << "}";
    }

    os << ",\"operators_computed\":";
    AppendU64List(os, record.operators_computed);
    // (slot-7 WP-3, stage 2 item 3) Always present ([E] sec:schema (a)).
    os << ",\"anemic_pool\":";
    AppendU64List(os, record.anemic_pool);
    // (slot-7 WP-3, stage 2 item 4) Always present ([E] sec:schema (b)).
    os << ",\"s_sep\":";
    if (record.s_sep.computed) {
        os << "{\"status\":\"computed\",\"primes\":";
        AppendU64List(os, record.s_sep.primes);
        os << ",\"added\":";
        AppendU64List(os, record.s_sep.added);
        os << "}";
    } else {
        os << "{\"status\":\"atlas_unavailable\"}";
    }

    if (record.has_phase_reached_high_water) {
        CheckPlain(record.phase_reached_high_water, "phase_reached_high_water");
        os << ",\"phase_reached_high_water\":\"" << record.phase_reached_high_water << "\"";
    }

    os << ",\"ideals\":[";
    for (size_t i = 0; i < record.ideals.size(); ++i) {
        if (i) os << ",";
        AppendIdeal(os, record.ideals[i], record, opts, seen);
    }
    os << "]";

    os << ",\"identities\":{\"passed\":" << record.identities_passed
       << ",\"failed\":" << record.identities_failed << "}";

    os << ",\"certificates_ran\":[";
    for (size_t i = 0; i < record.certificates_ran.size(); ++i) {
        CheckPlain(record.certificates_ran[i], "certificates_ran");
        if (i) os << ",";
        os << "\"" << record.certificates_ran[i] << "\"";
    }
    os << "]";

    os << ",\"hashes\":{";
    for (size_t i = 0; i < record.hashes.size(); ++i) {
        CheckPlain(record.hashes[i].first, "hash name");
        if (i) os << ",";
        os << "\"" << record.hashes[i].first << "\":\"" << oc_util::HexU64(record.hashes[i].second, false)
           << "\"";
    }
    os << "}";

    os << ",\"verify\":" << record.verify();
    os << "}";
}

std::string BuildHeckeExtReportJson(const HeckeExtRecord& record) {
    return BuildHeckeExtReportJson(record, ExtReportOptions());
}

std::string BuildHeckeExtReportJson(const HeckeExtRecord& record,
                                    const ExtReportOptions& opts) {
    std::ostringstream oss;
    StreamHeckeExtReportJson(oss, record, opts);
    return oss.str();
}

void WriteHeckeExtReport(const HeckeExtRecord& record, const std::string& path) {
    // R3.1 item 6: THE RECORD LAYER is the only place that knows the stem.
    // The sidecar is referenced by RELATIVE path (its basename), so the pair
    // of files moves together.
    std::string dir;
    std::string base = path;
    const size_t slash = path.find_last_of('/');
    if (slash != std::string::npos) {
        dir = path.substr(0, slash);
        base = path.substr(slash + 1);
    }
    const std::string kJson = ".json";
    std::string stem = base;
    if (stem.size() > kJson.size() &&
        stem.compare(stem.size() - kJson.size(), kJson.size(), kJson) == 0) {
        stem = stem.substr(0, stem.size() - kJson.size());
    }

    ExtReportOptions opts;
    opts.sidecar_stem = stem;
    opts.sidecar_dir = dir;

    std::ofstream out(path.c_str(), std::ios::binary);
    if (!out.good()) {
        throw std::runtime_error("HeckeExtReport: cannot open '" + path + "' for writing");
    }
    StreamHeckeExtReportJson(out, record, opts);
    out << "\n";
    out.close();
    if (!out.good()) {
        throw std::runtime_error("HeckeExtReport: error writing '" + path + "'");
    }
}

} // namespace hecke_ext
