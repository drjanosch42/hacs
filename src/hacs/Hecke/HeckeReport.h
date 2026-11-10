// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

#pragma once
#ifndef HECKE_REPORT_H
#define HECKE_REPORT_H

// Stage S11 -- Hecke/HeckeReport.{h,cpp}: hand-rolled JSON writer (no
// third-party library, plan Sec. 3 binding requirement) for the full
// three-matrix-level report of docs/implementation/hecke_action_implementation_plan.md
// Sec. 3 "Output formats", wrapping a COMPUTED hecke::HeckeModuleDecomposition
// (level (c)) together with metadata and the level (b) matrices
// A^{(q)}(T_p) for every computed prime p; an OPTIONAL, size-gated level (a)
// Theta^{(q)}_flat export.
//
// SCHEMA (binding; plan Sec. 3 JSON example + amendment A-N4, documented
// in-band in the emitted JSON and in Hecke/README.md):
//   * polynomial coefficient lists: LOW -> HIGH degree (constant term first),
//   * matrices: arrays of ROWS (row-major), entries canonical residues in
//     [0, ell),
//   * per level-(c) block: "residue_degree" (proven exact) XOR
//     "residue_degree_lower_bound" (proven lower bound only, NOT yet
//     certified equal to the true residue field degree) -- these two keys
//     are NEVER both present on the same block (see
//     HeckeModuleDecomposition.h for the full witness-search description);
//     "residue_degree_certified" (bool) makes the distinction
//     machine-readable without needing to know which key was used;
//     "residue_witness" records the certifying (or attempted) linear
//     combination sum c_p T_p for O(1) independent re-verification;
//
// SCHEMA v2 (hecke-extensions phase E0b, slot 2 WP-2 -- the stage's one
// golden re-baseline; [X] Sec. 1.6 / Sec. 3.1, [C] `sec:recordschema` item 5):
//   * "schema" is now "hecke_report/v2";
//   * RECORD LEVEL, the four oc1a fields, PER COMPLEX and never a maximum
//     over several ([C] `sec:recordschema` item 5 as revised in [C] v1.4.0;
//     slot-2 owner confirmation 1): "q_max" (the top degree of the complex
//     this record is about), "truncated" (was the complex cut off below its
//     natural top degree?), "q_max_source" ("derived" when the writer read
//     q_max off the complex, "set" when a setting fixed it -- E9's truncated
//     model is the first caller that will say "set"), and "degrees"
//     ([0 .. q_max], so that a per-degree record inside a truncated model is
//     self-describing). They are derived from HeckeReportInput::num_degrees,
//     the caller's own degree count -- NEVER literals;
//   * "level_c.locality": the `locality` mathematical setting in force
//     ([X] Sec. 1.5) with its source, plus the two scalars of the locality
//     loop's resumable state that say whether its certificate is complete.
//     The state ITSELF (LocalityState::to_json(), with its per-block bases
//     and candidate log) is R3 sidecar material, not report material;
//   * per level-(c) block, the locality certificate of [E] Cert. 5.2 as
//     [E] Sec. 7 names it: "local" (always present), and -- present exactly
//     when "local" is true, the A-N4 rule that an unproven claim is never
//     emitted under an unqualified key -- "witness_poly" (pi_w, low->high)
//     and "coefficient_field": {"w_hat_exponent": N, "P": {p -> P_p}}.
//     "P" IS AN OBJECT KEYED BY PRIME, not the single polynomial of [E]
//     Sec. 7: the character P_p is one polynomial PER OPERATOR of the family
//     (PrimaryBlock::character), and on a single-operator decomposition --
//     every decomposition the golden harness and the campaign build -- it has
//     exactly one entry. Emitting the map keeps the record honest for k > 1
//     and matches the "min_polys" / "char_polys" / "eigen_system" convention
//     of this very schema.
//   * "local" and "residue_degree_certified" are TWO DIFFERENT CLAIMS; see
//     HeckeModuleDecomposition.h.
//
//   * "eigen_system": present ONLY for a degree-1 block (dim == 1,
//     residue_degree == 1, certified) -- the map p -> theta_p, theta_p being
//     the scalar T_p acts by on that block (the Ash-Stevens eigenvalue
//     system value), read off the block's linear char_poly[p] = X - theta_p;
//   * a block with dim > residue_degree (certified) is an ISOTYPIC
//     T-primary block of multiplicity dim/residue_degree OVER THE COMPUTED
//     FAMILY -- documented in-band via "multiplicity_note", which since
//     hecke_report/v2 states three things and claims nothing more: the
//     multiplicity, whether the block is semisimple (every generalized-Jordan
//     partition (1,...,1)) and whether the family is a SINGLE operator, in
//     which case a larger commuting family may refine the block further.
//     (The pre-v2 note called such a block "genuinely irreducible ... NOT
//     further separable by more Hecke operators". Both halves are false: an
//     isotypic block of multiplicity m > 1 is not irreducible, and a block
//     computed from one operator is routinely refined by a second one.
//     [I G1]; corrected in hecke-extensions phase E0b, slot 2 WP-2.)
//
// This file builds the JSON purely by std::ostringstream string assembly
// (mirroring hecke::HeckeModuleDecomposition::to_json()'s existing style);
// it does not itself parse JSON (the round-trip mini-parser used by the S11
// Definition-of-Done smoke validation lives locally in
// HeckeSimultaneousSmokeValidation.cpp, in keeping with the codebase
// convention of per-file, from-scratch validation-only helper machinery).

#include "HeckeModuleDecomposition.h"

#include "../../ModArithmetic/FlatMatrix.h"
// (matrix-engine Phase 3) Matrix.h is no longer included directly: level_b's
// MatrixT is the A^{(q)} that HeckeModuleDecomposition (included above)
// consumes, and MatrixT itself arrives through FlatMatrix.h.

#include <cstdint>
#include <iosfwd>
#include <map>
#include <string>
#include <vector>

namespace hecke {

// ===========================================================================
// (matrix-engine Phase 6, audit item 7.19) STREAMING LEVEL (a).
//
// Level (a) is the (N_q*d_V)^2 matrix Theta^{(q)}_flat. At n_sym = 1000 that
// is 1.2e8 entries per degree, i.e. a multi-hundred-MB JSON array -- and the
// pre-Phase-6 writer built the WHOLE report in one std::ostringstream before
// anybody saw a byte of it, on top of a fully materialised FlatMatrix.
//
// A LevelARowSource is the pull interface that removes both: the writer asks
// for one row at a time and emits it immediately, so the peak cost of level
// (a) is one row (cols() entries) plus whatever the source itself needs. The
// campaign driver's implementation reads engine::BlockMat rows straight off
// HeckeOperatorOnK::Theta_block(q) with blk::materialise_rows, so the
// (N_q*d_V)^2 FlatMatrix is never built at all.
//
// The emitted BYTES are identical to the ones the resident path emits for the
// same matrix -- StreamHeckeReportJson has exactly one level-(a) writer, fed
// either from `level_a_theta_flat` or from `level_a_stream`.
// ===========================================================================
class LevelARowSource {
public:
    virtual ~LevelARowSource() = default;
    virtual uint32_t rows() const = 0;
    virtual uint32_t cols() const = 0;
    // Fills `out` with row `r` (canonical residues in [0, ell)); the callee
    // resizes `out` to cols(). Called with r = 0 .. rows()-1, in order.
    virtual void row(uint32_t r, std::vector<uint32_t>& out) const = 0;
};

// Everything needed to render one report. `decomp` must already have had
// compute() called on it (blocks()/change_of_basis()/certify() are read).
struct HeckeReportInput {
    uint32_t ell = 0;
    uint32_t n_sym = 0;
    uint32_t degree = 0; // the cohomological degree q the matrices below belong to

    // ---- (hecke-extensions phase E0b, slot 2 WP-2) THE oc1a FIELDS' INPUT.
    //
    // The number of cohomological degrees of THE COMPLEX this record is about
    // (q runs 0 .. num_degrees-1): the caller's own dims.size() /
    // get_H_matrices().size(). The writer DERIVES "q_max" = num_degrees - 1
    // and "degrees" = [0 .. q_max] from it, so a truncated model (E9) reports
    // its own top degree by construction and no literal `2` is ever written.
    // Required: the writer throws std::invalid_argument on 0, or on
    // `degree >= num_degrees`.
    uint32_t num_degrees = 0;

    // Was the complex cut off below its natural top degree? (E9 / [C]
    // `open:ell2`. false everywhere in this build: the 7-vertex bulk model is
    // complete in degrees 0..2 and the circle model in 0..1.)
    bool truncated = false;

    // Did q_max come from a SETTING rather than from the complex? Drives
    // "q_max_source": false -> "derived" (this build), true -> "set".
    bool q_max_configured = false;

    // Level (b): A^{(q)}(T_p) for every computed Hecke prime p (may be a
    // superset of the decomposition's own operator family, e.g. primes
    // computed for level (b) but not included in the level-(c) family).
    std::map<uint64_t, Matrix> level_b;

    // Level (c): the computed decomposition (compute() already called;
    // blocks() / change_of_basis() / certify() are consulted). Must outlive
    // the BuildHeckeReportJson() call.
    const HeckeModuleDecomposition* decomp = nullptr;

    // OPTIONAL, size-gated level (a): Theta^{(q)}_flat per computed p. Left
    // empty (the default, and the common case for anything but tiny models)
    // to omit level (a) from the report entirely -- these matrices are
    // N_q*d_V square and can be large; the caller decides the size gate.
    // (matrix-engine Phase 3) FlatMatrix, so a caller can hand over
    // get_Theta_flat_matrices()[q] without a (N_q*d_V)^2 `.to_matrix()` copy.
    std::map<uint64_t, FlatMatrix> level_a_theta_flat;

    // (matrix-engine Phase 6) The STREAMED form of the same level (a): if this
    // map is non-empty it is used and `level_a_theta_flat` must be empty (the
    // writer throws std::invalid_argument if both are populated). Pointers are
    // borrowed and must outlive the call.
    std::map<uint64_t, const LevelARowSource*> level_a_stream;
};

// Builds the full report JSON string (does not write to any file/stream;
// see WriteHeckeReport for that). Throws std::invalid_argument if
// input.decomp is null.
//
// (matrix-engine Phase 6) This is now a thin wrapper: it runs
// StreamHeckeReportJson into an std::ostringstream and returns the buffer, so
// the two entry points cannot drift and every frozen golden `report_full` /
// `report_canonical` hash keeps its value.
std::string BuildHeckeReportJson(const HeckeReportInput& input);

// (matrix-engine Phase 6, audit item 7.19) The same report, written straight
// to `os` without ever holding it in memory. Use it with an std::ofstream for
// large models. Throws std::invalid_argument if input.decomp is null or if
// both level-(a) sources are populated.
void StreamHeckeReportJson(std::ostream& os, const HeckeReportInput& input);

// Default report path, matching the plan's naming convention
// "hecke_report_ell<ell>_n<n>.json" (written to the working directory).
std::string DefaultHeckeReportPath(uint32_t ell, uint32_t n_sym);

// Writes `json` to `path` (creates/truncates the file); if `path` is empty,
// writes to stdout instead (plan Sec. 3: "written to a caller-specified path
// (default: stdout)"). Throws std::runtime_error on an I/O failure.
void WriteHeckeReport(const std::string& json, const std::string& path);

} // namespace hecke

#endif // HECKE_REPORT_H
