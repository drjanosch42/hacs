// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// hecke-extensions slot 5, WP-8 -- see HeckeM1Table.h for the M1 statement
// this file turns into rows and into a predicate.

#include "HeckeM1Table.h"

#include "HeckeExtFinal.h"
#include "HeckeSturmBound.h"

#include <sstream>

namespace hecke_ext {

namespace {

std::string Parts(const std::vector<uint32_t>& v) {
    if (v.empty()) return "-";
    std::ostringstream os;
    for (size_t i = 0; i < v.size(); ++i) {
        if (i) os << ",";
        os << v[i];
    }
    return os.str();
}

// The six enumerated `certificate` values of [E] Sec. 7 (four, plus dickson / boundary since slot-12 FIN).
bool IsCertificateValue(const std::string& s) {
    return s == "cyclic" || s == "commutant" || s == "sturm(B')" || s == "unavailable" ||
           s == "dickson" || s == "boundary";   // (slot-12 FIN) the finality certificates
}

} // namespace

// ===========================================================================
// The rows.
// ===========================================================================
std::vector<M1Row> M1RowsFromRecord(const HeckeExtRecord& record,
                                    const std::string& case_label) {
    // --- the cell-level columns, computed once. ---------------------------
    const uint32_t k = record.weight;
    bool     has_Bp = false;
    uint32_t Bp = 0, Bp_cap = 0;
    std::string Bp_note = "not_applicable";
    if (k >= 4 && k % 2 == 0 && SturmBound::dimension_M_k(k) > 0 && record.ell >= 2) {
        SturmBound bound(k, static_cast<uint32_t>(record.ell));
        bound.run();
        has_Bp = bound.has_B_prime();
        Bp = bound.B_prime();
        Bp_cap = bound.cap();
        Bp_note = has_Bp ? "found" : "none_at_cap";
    }

    uint32_t dim_H0 = 0, dim_H1 = 0;
    for (const GroupRecord& g : record.groups) {
        if (g.name == "H0" && g.available) dim_H0 = g.dim;
        if (g.name == "H1" && g.available) dim_H1 = g.dim;
    }

    std::vector<M1Row> rows;
    rows.reserve(record.ideals.size());
    for (const IdealRecord& id : record.ideals) {
        M1Row r;
        r.n = record.n_sym;
        r.ell = record.ell;
        r.weight = k;
        r.case_label = case_label;
        r.dim_H0 = dim_H0;
        r.dim_H1 = dim_H1;
        r.u_ell_included = record.u_ell.present && record.u_ell.included;
        r.u_ell_reason = record.u_ell.present ? record.u_ell.reason : std::string("unavailable");
        r.has_B_prime = has_Bp;
        r.B_prime = Bp;
        r.B_prime_cap = Bp_cap;
        r.B_prime_note = Bp_note;

        // The ambient group of the block: its OWN `groups` entry.
        r.group = "?";
        for (const IdealGroupMatch& gm : id.groups) {
            if (gm.own) { r.group = gm.group; break; }
        }
        if (r.group == "?" && !id.modules.empty()) r.group = id.modules[0].group;

        r.ideal_id = id.id;
        r.dim = id.dim;
        r.residue_degree = id.residue_degree;
        r.local = id.local;
        r.algebra_dim = id.algebra_dim;
        r.embdim = id.embdim;
        r.nilpotency = id.nilpotency;
        r.gorenstein = id.gorenstein;
        for (const ModuleRecordEntry& m : id.modules) {
            if (m.group == r.group) {
                r.loewy = m.structure.loewy;
                r.socle = m.structure.socle;
                r.cyclic = m.structure.cyclic;
                break;
            }
        }
        r.certificate = id.certificate;
        r.route_a = id.certificate_routes.a;
        r.route_b = id.certificate_routes.b;
        r.route_c = id.certificate_routes.c;
        r.route_d0 = id.certificate_routes.d0;
        r.route_d1 = id.certificate_routes.d1;
        r.final_value = ToString(id.final.value);
        r.final_reason = id.final.reason;
        rows.push_back(r);
    }
    return rows;
}

// ===========================================================================
// Well-formedness -- the M1 statement as a predicate.
// ===========================================================================
uint32_t CheckM1Row(const M1Row& row) {
    // 1: X_1 (+) X_4 and nothing else.
    if (row.group != "H0" && row.group != "H1") return 1;

    // 2: the tri-state.
    if (row.final_value != "true" && row.final_value != "false" &&
        row.final_value != "heuristic") {
        return 2;
    }

    // 3: THE fixed-list clause.
    if (row.final_value == "true") {
        if (!row.final_reason.empty()) return 3;
    } else {
        if (row.final_reason.empty()) return 3;
        if (!IsFinalReason(row.final_reason)) return 3;
    }

    // 4: B' on every row -- a value, or a note saying which of the two it is.
    if (!row.has_B_prime && row.B_prime_note != "none_at_cap" &&
        row.B_prime_note != "not_applicable") {
        return 4;
    }
    if (row.has_B_prime && (row.B_prime < 2 || row.B_prime_note != "found")) return 4;

    // 5: the certificate vocabulary, and "a certificate implies a route".
    if (!IsCertificateValue(row.certificate)) return 5;
    if (row.certificate != "unavailable" && row.route_a == "not_run" &&
        row.route_b == "not_run" && row.route_c == "not_run") {
        return 5;
    }
    // (slot-12 FIN) a finality certificate names the route that passed.
    if (row.certificate == "dickson" && row.route_d0 != "passed") return 5;
    if (row.certificate == "boundary" && row.route_d1 != "passed") return 5;

    // 6: finality without a route is the one combination M1 may not publish.
    if (row.final_value == "true" && row.certificate == "unavailable") return 6;

    return 0;
}

const char* M1RowMessage(uint32_t code) {
    switch (code) {
        case 0: return "ok";
        case 1: return "the ambient group is neither X_1 (H0) nor X_4 (H1)";
        case 2: return "`final` is not one of true / false / heuristic";
        case 3: return "the `final` reason is not a member of the fixed list of [X] R3.2";
        case 4: return "no B'(k, ell) column: neither a bound nor a note saying why not";
        case 5: return "the certificate string is unenumerated, or claimed with every route not_run";
        case 6: return "final = true with certificate = unavailable";
        default: return "unknown code";
    }
}

// ===========================================================================
// The printable table.
// ===========================================================================
std::string M1Header() {
    return "M1 n ell k group id dim f local A_S embdim nu gor loewy socle cyc "
           "cert a b c B' final reason u_ell case";
}

std::string M1RowText(const M1Row& row) {
    std::ostringstream os;
    os << "M1 " << row.n << " " << row.ell << " " << row.weight
       << " " << row.group << " " << row.ideal_id
       << " " << row.dim << " " << row.residue_degree
       << " " << (row.local ? "local" : "NOTLOCAL")
       << " " << row.algebra_dim << " " << row.embdim << " " << row.nilpotency
       << " " << (row.gorenstein ? "gor" : "-")
       << " " << Parts(row.loewy) << " " << Parts(row.socle)
       << " " << (row.cyclic ? "cyc" : "-")
       << " " << row.certificate
       << " " << row.route_a << " " << row.route_b << " " << row.route_c
       << " " << (row.has_B_prime ? std::to_string(row.B_prime) : row.B_prime_note)
       << " " << row.final_value
       << " " << (row.final_reason.empty() ? std::string("-") : row.final_reason)
       << " " << row.u_ell_reason
       << " " << row.case_label;
    return os.str();
}

} // namespace hecke_ext
