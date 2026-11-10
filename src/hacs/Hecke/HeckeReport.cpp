// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// Stage S11 -- see HeckeReport.h for the schema documentation.

#include "HeckeReport.h"

#include "FpPoly.h"
#include "HeckeSettings.h"

#include <fstream>
#include <iostream>
#include <ostream>
#include <sstream>
#include <stdexcept>
#include <vector>

namespace hecke {

namespace {

void EmitMatrixJson(std::ostream& oss, const Matrix& M) {
    oss << "[";
    for (uint32_t r = 0; r < M.rows; ++r) {
        if (r > 0) oss << ",";
        oss << "[";
        for (uint32_t c = 0; c < M.cols; ++c) {
            if (c > 0) oss << ",";
            oss << M.data[r][c];
        }
        oss << "]";
    }
    oss << "]";
}

// Same output shape as EmitMatrixJson, for a FlatMatrix (level (a)).
void EmitFlatMatrixJson(std::ostream& oss, const FlatMatrix& M) {
    oss << "[";
    for (uint32_t r = 0; r < M.get_rows(); ++r) {
        if (r > 0) oss << ",";
        oss << "[";
        for (uint32_t c = 0; c < M.get_cols(); ++c) {
            if (c > 0) oss << ",";
            oss << M.get_entry(r, c);
        }
        oss << "]";
    }
    oss << "]";
}

// (Phase 6) Byte-for-byte the same output as EmitFlatMatrixJson, one row at
// a time -- the only difference is where the row comes from.
void EmitRowSourceJson(std::ostream& oss, const LevelARowSource& src) {
    std::vector<uint32_t> row;
    oss << "[";
    for (uint32_t r = 0; r < src.rows(); ++r) {
        if (r > 0) oss << ",";
        src.row(r, row);
        if (row.size() != src.cols()) {
            throw std::runtime_error("HeckeReport: LevelARowSource returned a row of the wrong width");
        }
        oss << "[";
        for (uint32_t c = 0; c < src.cols(); ++c) {
            if (c > 0) oss << ",";
            oss << row[c];
        }
        oss << "]";
    }
    oss << "]";
}

void EmitPolyJson(std::ostream& oss, const FpPoly& P) {
    // LOW -> HIGH degree (A-N4).
    oss << "[";
    for (size_t i = 0; i < P.coeffs.size(); ++i) {
        if (i > 0) oss << ",";
        oss << P.coeffs[i];
    }
    oss << "]";
}

// theta such that the (monic, degree-1) polynomial equals X - theta:
// coeffs = [c0, 1] (low -> high), so theta = -c0 mod ell.
uint32_t LinearRoot(const FpPoly& linear, uint32_t ell) {
    if (linear.degree() != 1) {
        throw std::invalid_argument("HeckeReport: LinearRoot needs a degree-1 polynomial");
    }
    const uint32_t c0 = linear.coeffs[0];
    return (c0 == 0) ? 0u : (ell - c0);
}

// (hecke-extensions phase E0b, slot 2 WP-2) A block is SEMISIMPLE iff every
// generalized-Jordan partition, for every operator of the family, is
// (1, 1, ..., 1) -- equivalently every min-poly exponent e is 1. One of the
// three things the corrected `multiplicity_note` states.
bool BlockIsSemisimple(const PrimaryBlock& b) {
    for (const auto& kv : b.jordan_partition) {
        for (uint32_t part : kv.second) {
            if (part != 1) return false;
        }
    }
    return true;
}

} // namespace

std::string BuildHeckeReportJson(const HeckeReportInput& input) {
    // (Phase 6) One writer, two entry points -- see the header note.
    std::ostringstream oss;
    StreamHeckeReportJson(oss, input);
    return oss.str();
}

void StreamHeckeReportJson(std::ostream& oss, const HeckeReportInput& input) {
    // (slot-8 WP-3) [E] sec:ring (b): hecke_report/v2 never moves (surface).
    RequireFieldCoefficientRing("HeckeReport (hecke_report/v2)");
    if (input.decomp == nullptr) {
        throw std::invalid_argument("HeckeReport: HeckeReportInput.decomp must not be null");
    }
    if (!input.level_a_theta_flat.empty() && !input.level_a_stream.empty()) {
        throw std::invalid_argument(
            "HeckeReport: level_a_theta_flat and level_a_stream are mutually exclusive");
    }
    const HeckeModuleDecomposition& decomp = *input.decomp;
    const std::vector<PrimaryBlock>& blocks = decomp.blocks();

    // (E0b, slot 2 WP-2) The oc1a fields are DERIVED from the caller's own
    // degree count -- see HeckeReport.h. A record that cannot name the complex
    // it is about is refused rather than guessed.
    if (input.num_degrees == 0) {
        throw std::invalid_argument(
            "HeckeReport: HeckeReportInput.num_degrees must be the complex's degree count "
            "(q_max, truncated, q_max_source and degrees are derived from it)");
    }
    if (input.degree >= input.num_degrees) {
        throw std::invalid_argument(
            "HeckeReport: HeckeReportInput.degree is outside the complex's degree range "
            "[0, num_degrees)");
    }
    const uint32_t q_max = input.num_degrees - 1u;

    oss << "{";
    oss << "\"schema\":\"hecke_report/v2\",";
    oss << "\"ell\":" << input.ell << ",";
    oss << "\"n_sym\":" << input.n_sym << ",";
    oss << "\"degree\":" << input.degree << ",";
    // The four per-complex oc1a fields ([C] `sec:recordschema` item 5, v1.4.0;
    // slot-2 owner confirmation 1: PER COMPLEX, never a maximum over a record).
    oss << "\"q_max\":" << q_max << ",";
    oss << "\"truncated\":" << (input.truncated ? "true" : "false") << ",";
    oss << "\"q_max_source\":\"" << (input.q_max_configured ? "set" : "derived") << "\",";
    oss << "\"degrees\":[";
    for (uint32_t q = 0; q <= q_max; ++q) {
        if (q > 0) oss << ",";
        oss << q;
    }
    oss << "],";
    oss << "\"poly_coeff_order\":\"low->high degree (A-N4)\",";
    oss << "\"matrix_entry_order\":\"arrays of rows (row-major); entries are "
           "canonical residues in [0, ell)\",";

    // "operators": the union of level_b's primes and the level-(c) family's
    // primes, ascending (std::map already sorts by key -- deterministic).
    std::map<uint64_t, bool> all_primes;
    for (const auto& kv : input.level_b) all_primes[kv.first] = true;
    for (const PrimaryBlock& b : blocks) {
        for (const auto& kv : b.char_poly) all_primes[kv.first] = true;
    }
    oss << "\"operators\":[";
    {
        bool first = true;
        for (const auto& kv : all_primes) {
            if (!first) oss << ",";
            first = false;
            oss << "{\"p\":" << kv.first << "}";
        }
    }
    oss << "],";

    // Level (a) -- OPTIONAL, size-gated by the caller (empty map => omitted).
    oss << "\"level_a\":";
    if (input.level_a_theta_flat.empty() && input.level_a_stream.empty()) {
        oss << "null,";
    } else if (!input.level_a_stream.empty()) {
        // (Phase 6) Streamed: identical bytes, one row resident at a time.
        oss << "{";
        bool first = true;
        for (const auto& kv : input.level_a_stream) {
            if (!first) oss << ",";
            first = false;
            if (kv.second == nullptr) {
                throw std::invalid_argument("HeckeReport: null LevelARowSource in level_a_stream");
            }
            oss << "\"" << kv.first << "\":";
            EmitRowSourceJson(oss, *kv.second);
        }
        oss << "},";
    } else {
        oss << "{";
        bool first = true;
        for (const auto& kv : input.level_a_theta_flat) {
            if (!first) oss << ",";
            first = false;
            oss << "\"" << kv.first << "\":";
            EmitFlatMatrixJson(oss, kv.second);
        }
        oss << "},";
    }

    // Level (b): A^{(q)}(T_p) for every computed p.
    oss << "\"level_b\":{";
    {
        bool first = true;
        for (const auto& kv : input.level_b) {
            if (!first) oss << ",";
            first = false;
            oss << "\"" << kv.first << "\":";
            EmitMatrixJson(oss, kv.second);
        }
    }
    oss << "},";

    // Level (c): the primary decomposition / MaxSpec(T) blocks.
    uint32_t cohomology_dim = 0;
    for (const PrimaryBlock& b : blocks) cohomology_dim += b.dim;

    // (E0b, slot 2 WP-2) The family size, one of the three things the
    // corrected `multiplicity_note` states.
    const size_t n_ops = decomp.ops().size();

    oss << "\"level_c\":{";
    oss << "\"cohomology_dim\":" << cohomology_dim << ",";
    // (E0b, slot 2 WP-2) `level_c.locality`: the `locality` MATHEMATICAL
    // setting in force ([X] Sec. 1.5) WITH ITS SOURCE -- recorded, never
    // inferred -- plus the two scalars of the loop's resumable state that say
    // whether the certificate under that budget is complete. The state itself
    // (LocalityState::to_json(), per-block bases + candidate log) belongs in
    // an R3 sidecar, not in every report.
    const bool strict_locality = (math_settings().locality.mode == LocalityMode::Strict);
    {
        const MathSettings& ms = math_settings();
        const LocalityState& ls = decomp.locality_state();
        oss << "\"locality\":{\"mode\":\""
            << (ms.locality.mode == LocalityMode::Strict ? "strict" : "budget")
            << "\",\"budget\":" << ms.locality.budget
            << ",\"source\":\"" << to_string(ms.locality_source)
            << "\",\"complete\":" << (ls.complete ? "true" : "false")
            << ",\"splits\":" << ls.splits;
        // (E0b-S, slot 3 WP-3) `phase_reached` is emitted ONLY under strict
        // mode ([X] Sec. 3.1 step 4): under the default `budget=<N>` there is
        // only one phase, and the default report text must not move, since it
        // is what the golden `report_*` objects hash.
        if (ms.locality.mode == LocalityMode::Strict) {
            oss << ",\"phase_reached\":\"" << to_string(ls.phase_reached) << "\""
                << ",\"interrupted\":" << (ls.interrupted ? "true" : "false");
        }
        oss << "},";
    }
    oss << "\"blocks\":[";
    for (size_t j = 0; j < blocks.size(); ++j) {
        const PrimaryBlock& b = blocks[j];
        if (j > 0) oss << ",";
        oss << "{\"dim\":" << b.dim << ",";

        // A-N4: an unproven residue degree is NEVER emitted under the
        // unqualified "residue_degree" key.
        if (b.residue_degree_certified) {
            oss << "\"residue_degree\":" << b.residue_degree << ",";
        } else {
            oss << "\"residue_degree_lower_bound\":" << b.residue_degree << ",";
        }
        oss << "\"residue_degree_certified\":"
            << (b.residue_degree_certified ? "true" : "false") << ",";

        // (E0b, slot 2 WP-2) THE LOCALITY CERTIFICATE ([E] Cert. 5.2, key
        // names of [E] Sec. 7). "local" is the claim "this block is ONE
        // maximal ideal of T" -- a different claim from
        // "residue_degree_certified"; see HeckeModuleDecomposition.h.
        oss << "\"local\":" << (b.local_certified ? "true" : "false") << ",";
        // (E0b-S, slot 3 WP-3) The phase in which THIS block's verdict was
        // reached -- strict-only, exactly like the level_c.locality subkey.
        if (strict_locality) {
            oss << "\"phase_reached\":\"" << to_string(b.phase_reached) << "\",";
        }

        // (E0b, slot 2 WP-2) THE CORRECTED multiplicity_note ([I G1], [X]
        // Sec. 3.1). It states three things and claims nothing beyond them:
        // the multiplicity, semisimplicity, and whether the family is a
        // single operator. The pre-v2 text called an isotypic block
        // "genuinely irreducible ... NOT further separable by more Hecke
        // operators"; an isotypic block of multiplicity m > 1 is not
        // irreducible, and a block computed from ONE operator is routinely
        // refined by a second one.
        oss << "\"multiplicity_note\":\"";
        if (b.residue_degree_certified && b.residue_degree > 0 &&
            b.dim % b.residue_degree == 0 && b.dim > b.residue_degree) {
            oss << "isotypic multiplicity " << (b.dim / b.residue_degree)
                << " over the computed operator family (residue field degree "
                << b.residue_degree << ")";
        } else if (b.residue_degree_certified) {
            oss << "multiplicity 1 (a single Galois-conjugacy class of eigenvalue systems, "
                   "residue field degree " << b.residue_degree << ")";
        } else {
            oss << "multiplicity not determined -- residue degree not yet certified "
                   "(see residue_degree_lower_bound)";
        }
        oss << "; " << (BlockIsSemisimple(b)
                            ? "semisimple (every generalized-Jordan partition is (1,...,1))"
                            : "not semisimple (some generalized-Jordan partition has a part > 1)");
        if (n_ops == 1) {
            oss << "; computed from a SINGLE operator, so a larger commuting family may refine "
                   "this block further";
        } else {
            oss << "; computed from a commuting family of " << n_ops
                << " operators, which no operator of that family refines further -- an operator "
                   "outside it still may";
        }
        oss << "\",";

        oss << "\"residue_witness\":[";
        for (size_t t = 0; t < b.residue_witness.size(); ++t) {
            if (t > 0) oss << ",";
            oss << "{\"p\":" << b.residue_witness[t].first
                << ",\"c\":" << b.residue_witness[t].second << "}";
        }
        oss << "],";

        // (E0b, slot 2 WP-2) The certificate's data, emitted EXACTLY when
        // `local` is true -- the A-N4 rule that an unproven claim never
        // appears under an unqualified key (as for residue_degree). "P" is
        // keyed by prime: the character P_p is one polynomial per operator
        // (see HeckeReport.h).
        if (b.local_certified) {
            oss << "\"witness_poly\":";
            EmitPolyJson(oss, b.witness_poly);
            oss << ",\"coefficient_field\":{\"w_hat_exponent\":" << b.w_hat_exponent
                << ",\"P\":{";
            bool first_chi = true;
            for (const auto& kv : b.character) {
                if (!first_chi) oss << ",";
                first_chi = false;
                oss << "\"" << kv.first << "\":";
                EmitPolyJson(oss, kv.second);
            }
            oss << "}},";
        }

        oss << "\"min_polys\":{";
        {
            bool first = true;
            for (const auto& kv : b.min_poly) {
                if (!first) oss << ",";
                first = false;
                oss << "\"" << kv.first << "\":";
                EmitPolyJson(oss, kv.second);
            }
        }
        oss << "},\"char_polys\":{";
        {
            bool first = true;
            for (const auto& kv : b.char_poly) {
                if (!first) oss << ",";
                first = false;
                oss << "\"" << kv.first << "\":";
                EmitPolyJson(oss, kv.second);
            }
        }
        oss << "},\"jordan_partitions\":{";
        {
            bool first = true;
            for (const auto& kv : b.jordan_partition) {
                if (!first) oss << ",";
                first = false;
                oss << "\"" << kv.first << "\":[";
                for (size_t t = 0; t < kv.second.size(); ++t) {
                    if (t > 0) oss << ",";
                    oss << kv.second[t];
                }
                oss << "]";
            }
        }
        oss << "}";

        // "eigen_system": ONLY for a degree-1 block (dim == 1, residue
        // degree 1, certified) -- the Ash-Stevens {theta_p} value list.
        if (b.dim == 1 && b.residue_degree_certified && b.residue_degree == 1) {
            oss << ",\"eigen_system\":{";
            bool first = true;
            for (const auto& kv : b.char_poly) {
                if (!first) oss << ",";
                first = false;
                const uint32_t theta = LinearRoot(kv.second, input.ell);
                oss << "\"" << kv.first << "\":" << theta;
            }
            oss << "}";
        }

        oss << ",\"basis_columns\":";
        EmitMatrixJson(oss, b.basis_columns);
        oss << "}";
    }
    oss << "],";
    oss << "\"change_of_basis\":";
    EmitMatrixJson(oss, decomp.change_of_basis());
    oss << ",\"certified\":" << (decomp.certify() == 0 ? "true" : "false");
    oss << "}"; // level_c

    oss << "}"; // top-level object
}

std::string DefaultHeckeReportPath(uint32_t ell, uint32_t n_sym) {
    return "hecke_report_ell" + std::to_string(ell) + "_n" + std::to_string(n_sym) + ".json";
}

void WriteHeckeReport(const std::string& json, const std::string& path) {
    if (path.empty()) {
        std::cout << json << std::endl;
        return;
    }
    std::ofstream out(path, std::ios::out | std::ios::trunc);
    if (!out) {
        throw std::runtime_error("HeckeReport: failed to open '" + path + "' for writing");
    }
    out << json;
    if (!out) {
        throw std::runtime_error("HeckeReport: failed while writing '" + path + "'");
    }
}

} // namespace hecke
