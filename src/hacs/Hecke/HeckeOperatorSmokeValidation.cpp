// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// Stage S7 -- see HeckeOperatorSmokeValidation.h.

#include "HeckeOperatorSmokeValidation.h"

#include "HeckeCosetTables.h"
#include "HeckeOperatorOnK.h"
#include "HeckeTauTable.h"
#include "SL2ZTessellationTables.h"
#include "../EquivariantCohomologyOnK.h"
#include "../LocalSystemSL2Z.h"
#include "../LocalSystemSimplicialComplex.h"
#include "../SimplicialCohomology.h"

#include <cctype>
#include <iostream>
#include <string>
#include <vector>

namespace {

// ---------------------------------------------------------------------------
// Minimal embedded JSON structure parser (validation-only; accepts objects,
// arrays, strings without escapes-beyond-\", numbers, true/false/null).
// Returns true iff the whole input is exactly one valid JSON value.
// ---------------------------------------------------------------------------

struct JsonParser {
    const std::string& s;
    size_t pos = 0;

    explicit JsonParser(const std::string& text) : s(text) {}

    void skip_ws() {
        while (pos < s.size() && std::isspace(static_cast<unsigned char>(s[pos]))) ++pos;
    }
    bool parse_value() {
        skip_ws();
        if (pos >= s.size()) return false;
        const char c = s[pos];
        if (c == '{') return parse_object();
        if (c == '[') return parse_array();
        if (c == '"') return parse_string();
        if (c == 't') return parse_literal("true");
        if (c == 'f') return parse_literal("false");
        if (c == 'n') return parse_literal("null");
        return parse_number();
    }
    bool parse_literal(const char* lit) {
        const size_t n = std::string(lit).size();
        if (s.compare(pos, n, lit) != 0) return false;
        pos += n;
        return true;
    }
    bool parse_string() {
        if (s[pos] != '"') return false;
        ++pos;
        while (pos < s.size() && s[pos] != '"') {
            if (s[pos] == '\\') ++pos; // skip escaped char
            ++pos;
        }
        if (pos >= s.size()) return false;
        ++pos; // closing quote
        return true;
    }
    bool parse_number() {
        const size_t start = pos;
        if (pos < s.size() && (s[pos] == '-' || s[pos] == '+')) ++pos;
        while (pos < s.size() &&
               (std::isdigit(static_cast<unsigned char>(s[pos])) || s[pos] == '.' ||
                s[pos] == 'e' || s[pos] == 'E' || s[pos] == '-' || s[pos] == '+')) ++pos;
        return pos > start;
    }
    bool parse_object() {
        ++pos; // '{'
        skip_ws();
        if (pos < s.size() && s[pos] == '}') { ++pos; return true; }
        while (true) {
            skip_ws();
            if (pos >= s.size() || !parse_string()) return false; // key
            skip_ws();
            if (pos >= s.size() || s[pos] != ':') return false;
            ++pos;
            if (!parse_value()) return false;
            skip_ws();
            if (pos < s.size() && s[pos] == ',') { ++pos; continue; }
            if (pos < s.size() && s[pos] == '}') { ++pos; return true; }
            return false;
        }
    }
    bool parse_array() {
        ++pos; // '['
        skip_ws();
        if (pos < s.size() && s[pos] == ']') { ++pos; return true; }
        while (true) {
            if (!parse_value()) return false;
            skip_ws();
            if (pos < s.size() && s[pos] == ',') { ++pos; continue; }
            if (pos < s.size() && s[pos] == ']') { ++pos; return true; }
            return false;
        }
    }
    bool parse_document() {
        if (!parse_value()) return false;
        skip_ws();
        return pos == s.size();
    }
};

// ---------------------------------------------------------------------------
// One smoke case: builds the whole stack for (p_hecke=2, r=1, ell, n_sym),
// runs compute() (which enforces the runtime assertions and the O12(a)
// matrix-level bit-identity), and checks the expected 1x1 cohomology matrix
// A^{(q_expected)} = (a_expected) plus the surrounding structure.
// Returns 0 on success; nonzero codes localize the failure.
// ---------------------------------------------------------------------------

uint32_t run_case(const SimplicialComplex& K, const hecke::SL2ZTessellationTables& model,
                  uint32_t ell, uint32_t n_sym, uint32_t q_expected, uint32_t a_expected,
                  std::string& json_out) {
    const uint32_t d_V = n_sym + 1;

    // Base system L = Sym^n(F_ell^2) on the 7-vertex model (relations derived
    // from the S2 group data, invariant I3).
    std::vector<LocalSystemSimplicialComplex::SimplexRelation> rels;
    for (const auto& datum : GenerateSL2ZRelationsWithGroupData(K, n_sym, ell)) {
        rels.push_back(datum.rel);
    }
    LocalSystemSimplicialComplex lssc(K, rels, d_V, ell);
    EquivariantCohomologyOnK coh(&lssc);
    coh.compute_cohomology();

    // S3 + S4 tables for T_2 (g = diag(1, 2), 3 cosets).
    const hecke::HeckeCosetTables cosets(2, 1, ell, n_sym);
    const hecke::HeckeTauTable tau(K, model, cosets, ell, n_sym);

    hecke::HeckeOperatorOnK op(&coh, &cosets, &tau);
    op.compute(); // throws on ANY runtime-assertion or O12(a) failure

    // Level shapes: 3 degrees everywhere; Theta_flat is (N_q*d_V)^2;
    // A^{(q)} is d_q x d_q with 0 x 0 for empty degrees (no crash).
    const auto& theta_flat = op.get_Theta_flat_matrices();
    const auto& theta_eq = op.get_Theta_eq_matrices();
    const auto& A = op.get_H_matrices();
    if (theta_flat.size() != 3 || theta_eq.size() != 3 || A.size() != 3) return 10;
    const auto& H_dims = coh.get_H_eq_dimensions();
    for (uint32_t q = 0; q < 3; ++q) {
        const uint32_t dim = K.get_num_q_simplices(q) * d_V;
        if (theta_flat[q].get_rows() != dim || theta_flat[q].get_cols() != dim) return 11;
        if (A[q].rows != H_dims[q] || A[q].cols != H_dims[q]) return 12;
        if (theta_eq[q].get_rows() != theta_eq[q].get_cols()) return 13;
    }

    // The DoD oracle value: A^{(q_expected)} is the 1x1 matrix (a_expected).
    if (H_dims[q_expected] != 1) {
        std::cout << "HeckeOperatorSmokeValidation: dim H^" << q_expected << " = "
                  << H_dims[q_expected] << " != 1 for (ell,n)=(" << ell << ","
                  << n_sym << ")" << std::endl;
        return 20;
    }
    if (A[q_expected].data[0][0] != a_expected) {
        std::cout << "HeckeOperatorSmokeValidation: A^(" << q_expected << ") = ("
                  << A[q_expected].data[0][0] << ") != (" << a_expected
                  << ") for (ell,n)=(" << ell << "," << n_sym << ")" << std::endl;
        return 21;
    }

    // Route agreement on FULL standard bases of C^q(K;V), q = 0,1,2: fused
    // apply_Theta vs. the matrix-free direct route apply_route2 must be
    // bit-identical (this re-exercises O12(a) through the public apply API).
    for (uint32_t q = 0; q < 3; ++q) {
        const uint32_t dim = K.get_num_q_simplices(q) * d_V;
        for (uint32_t b = 0; b < dim; ++b) {
            FlatMatrix e_b(dim, 1, ell);
            e_b.set_entry(b, 0, 1);
            const FlatMatrix y1 = op.apply_Theta(q, e_b);
            const FlatMatrix y2 = op.apply_route2(q, e_b);
            for (uint32_t r = 0; r < dim; ++r) {
                if (y1.get_entry(r, 0) != y2.get_entry(r, 0)) {
                    std::cout << "HeckeOperatorSmokeValidation: routes differ at q="
                              << q << ", basis " << b << ", row " << r << std::endl;
                    return 30;
                }
            }
        }
    }

    json_out = op.to_json();
    return 0;
}

// Extracts the decimal integer following `key` (e.g. "\"relation_checks\":")
// from the JSON string; returns -1 if absent/malformed.
int64_t json_uint_after(const std::string& json, const std::string& key) {
    const size_t at = json.find(key);
    if (at == std::string::npos) return -1;
    size_t p = at + key.size();
    if (p >= json.size() || !std::isdigit(static_cast<unsigned char>(json[p]))) return -1;
    int64_t v = 0;
    while (p < json.size() && std::isdigit(static_cast<unsigned char>(json[p]))) {
        v = v * 10 + (json[p] - '0');
        ++p;
    }
    return v;
}

} // namespace

uint32_t HeckeOperatorSmokeValidation() {
    try {
        const SimplicialComplex K = CreateSL2ZFundamentalDomainTriangulation();
        const hecke::SL2ZTessellationTables model(K);

        // Case 1 (oracle O1 pin): (p, ell, n) = (2, 5, 0), A^{(0)} = (3).
        std::string json0;
        uint32_t code = run_case(K, model, /*ell=*/5, /*n_sym=*/0,
                                 /*q_expected=*/0, /*a_expected=*/3, json0);
        if (code != 0) return 100 + code;

        // Case 2 (oracle O3 pin): (p, ell, n) = (2, 5, 2), A^{(1)} = (4).
        std::string json2;
        code = run_case(K, model, /*ell=*/5, /*n_sym=*/2,
                        /*q_expected=*/1, /*a_expected=*/4, json2);
        if (code != 0) return 200 + code;

        // to_json(): fully parseable (embedded parser) ...
        for (const std::string* js : { &json0, &json2 }) {
            JsonParser parser(*js);
            if (!parser.parse_document()) {
                std::cout << "HeckeOperatorSmokeValidation: to_json() output is not "
                             "parseable JSON" << std::endl;
                return 300;
            }
        }
        // ... and carries the expected values and the active-assertion
        // certificate (nonzero check counters prove the runtime assertions
        // (i)/(ii)/(iii) actually executed inside compute(); residual == 0
        // is amendment A-N3's exact-solve certificate datum).
        if (json0.find("\"A\":[[3]]") == std::string::npos) return 301;
        if (json2.find("\"A\":[[4]]") == std::string::npos) return 302;
        for (const std::string* js : { &json0, &json2 }) {
            if (js->find("\"routes_bit_identical\":true") == std::string::npos) return 303;
            if (json_uint_after(*js, "\"relation_checks\":") < 1) return 304;
            if (json_uint_after(*js, "\"cocycle_checks\":") < 1) return 305;
            if (json_uint_after(*js, "\"theta_eq_solves\":") < 1) return 306;
            if (json_uint_after(*js, "\"theta_eq_residual_max\":") != 0) return 307;
            if (js->find("\"ell\":5") == std::string::npos) return 308;
            if (js->find("\"p_hecke\":2") == std::string::npos) return 309;
        }

        std::cout << "HeckeOperatorSmokeValidation: PASSED (A^(0)=(3) at (2,5,0), "
                     "A^(1)=(4) at (2,5,2), routes bit-identical, runtime "
                     "assertions executed, JSON parseable)." << std::endl;
        return 0;
    } catch (const std::exception& e) {
        std::cout << "HeckeOperatorSmokeValidation: EXCEPTION: " << e.what() << std::endl;
        return 1;
    }
}
