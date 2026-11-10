// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

#pragma once
#ifndef GOLDEN_HARNESS_H
#define GOLDEN_HARNESS_H

#include <cstdint>

// Phase 0 -- golden-output regression harness (docs/matrix-engine/
// 00-implementation-plan.md, "Phase 0" / "Golden harness", Sec. 4.1, and
// owner decision Q1 = "only canonical objects").
//
// Two registered validations, both driven by the SAME point grid and the SAME
// object extraction code (GoldenHarness.cpp):
//
//   GoldenRegressionValidation
//       Recomputes every golden object on the current binary and compares it
//       against the checked-in reference GoldenReference.inc.
//         * Tier A (acceptance; basis-independent, must never change):
//           any mismatch or missing/extra entry FAILS (nonzero return).
//         * Tier B (change detector; basis-dependent hashes): mismatches are
//           reported in full but only FAIL when the environment variable
//           GOLDEN_STRICT=1 is set. A tier-B change requires a documented,
//           object-by-object re-baseline (owner-approved), never a silent one.
//       Returns the number of failing entries (0 = success).
//
//   GoldenEmit
//       Recomputes every golden object and OVERWRITES GoldenReference.inc
//       (path derived from this source file's location, override with the
//       environment variable GOLDEN_OUT=<path>). Prints a loud warning: this
//       replaces the acceptance reference. SAFETY INTERLOCK: it only emits when
//       GOLDEN_EMIT=1 is set in the environment (so `--all`, which runs every
//       registered name, cannot rewrite the reference); without it, it prints
//       the instructions and returns 0. One-command regeneration:
//           GOLDEN_EMIT=1 ./cohomology_app GoldenEmit && make
//       Returns 0 on success.
//
// See the header comment of GoldenHarness.cpp for the grid, the per-object
// tier classification, the hash definition, and the trimming decision.
uint32_t GoldenRegressionValidation();
uint32_t GoldenEmit();

// (matrix-engine Phase 4) A SUBSET of the 32-bit grid for in-process
// re-runs: every golden object of the listed (n_sym, ell) cells is recomputed
// with the SAME extraction code as GoldenRegressionValidation and compared to
// the checked-in reference with BOTH tiers gating.  Returns the number of
// mismatching / missing entries and, through `digest`, one FNV-1a-64 over
// every (point, object, dims, hash) record in emission order -- so two calls
// under different engine settings (thread count, batching, ...) can be
// compared to each other AND to the reference.  `quiet` suppresses the
// per-cell progress lines.  Used by EngineDeterminismValidation.
#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>
uint32_t GoldenSubset32(const std::vector<std::pair<uint32_t, uint32_t>>& cells, uint64_t& digest,
                        size_t& records, bool quiet);

// ===========================================================================
// (hecke-extensions phase E0a) GoldenDump -- the "before" half of the E0b
// re-baseline protocol ([X] Sec. 1.6 steps 1-4).
// ===========================================================================
//
// GoldenReference.inc stores HASHES, not texts, so "explain every changed line
// object by object" (CONTRIBUTING.md, GoldenEmit) needs the old texts.  GoldenDump
// writes them: every TEXT object the harness hashes (report_full^q,
// report_canonical^q, op_json_full, and any `excluded` message) and every
// decomposition's BLOCK LIST -- per block dim, residue_degree,
// residue_degree_certified, residue_witness, per-operator char_poly / min_poly
// / jordan_partition, and the basis columns, i.e. exactly the inputs of the
// four hashes _invariants / _certify / _witness / _U -- plus certify().
//
// It is OWNER-GATED exactly like GoldenEmit: a no-op (exit 0, nothing written)
// unless GOLDEN_DUMP=<dir> is set in the environment, so `--all` -- which runs
// every registered name -- stays read-only and writes nothing.  Usage:
//     GOLDEN_DUMP=/tmp/before ./cohomology_app GoldenDump
// It writes <dir>/index.tsv, <dir>/texts.tsv and <dir>/decompositions.txt;
// docs/matrix-engine/tools/golden_explain.py diffs two such directories.
// GoldenDump computes exactly what GoldenRegressionValidation computes, through
// the same extraction code, and asserts nothing.
uint32_t GoldenDump();

// ===========================================================================
// (hecke-extensions phase E0a) The read-only locality sweep ([X] Sec. 3.3).
// ===========================================================================
//
// Recomputes EVERY decomposition this harness builds -- decomp_A(T_p)^q over
// the whole 32-bit grid and the multi-operator decomp_sim{2,3,5}^1 -- runs
// hecke::CertifyLocality on every block, and re-checks all FOUR golden
// decomposition hashes (_invariants, _certify, _U, _witness) against
// GoldenReference.inc through a lookup, never a copied constant.  This is what
// decides, BEFORE the reference is touched, whether E0b's fix can move a
// decomposition hash: if every block is local, [X] Sec. 3.3's Claim says none
// of the four moves and E0b's re-baseline is exactly the allow-listed report
// text.  Driven by HeckeLocalityValidation's grid arm.
struct GoldenLocalitySweepResult {
    uint64_t decompositions = 0;    // decompositions visited
    uint64_t blocks = 0;            // blocks certified
    uint64_t analytic_blocks = 0;   // of those, k = 1 (local for free)
    uint64_t nonlocal_blocks = 0;   // blocks CertifyLocality rejects
    uint64_t witness_changes = 0;   // local, but a different first witness
    // (E0b, slot 2 WP-1) compute()'s OWN locality certificate -- the one the
    // refinement loop now enforces -- disagreeing with the read-only verdict
    // of CertifyLocality on the same block.  It must be 0: the two are the
    // same candidate sequence and the same clause-(2)/(3) implementation.
    uint64_t loop_disagreements = 0;
    // (E0b-S, slot 3 WP-3) Blocks whose verdict was NOT reached in the linear
    // witness phase.  Under `locality = strict` the grid arm asserts this is
    // 0: strict mode enters its terminating route on no golden block, which is
    // why the two modes are byte-identical there.
    uint64_t nonlinear_phase_blocks = 0;
    uint64_t hashes_checked = 0;    // 4 per decomposition
    uint64_t hashes_failing = 0;
    std::vector<std::string> findings;      // one line per non-local block
    std::vector<std::string> hash_failures; // one line per mismatching hash
};
GoldenLocalitySweepResult GoldenLocalitySweep(bool quiet);

// ===========================================================================
// (console phase C1a) GoldenEntry / GoldenLookup -- THE read-only lookup into
// the compiled-in reference (console plan v3 Sec. 2 C5 and Sec. 10 "Phase C1a",
// inventory gap G13; [X] Sec. 1.1 "one lookup, not two").
// ===========================================================================
//
// GoldenReference.inc is a brace-initialiser list of these entries and is
// `#include`d in exactly one place, GoldenHarness.cpp.  Before C1a the lookup
// over it was a file-local `FindReference` used by GoldenLocalitySweep; C1a
// PROMOTES that one function to the public surface and re-routes the sweep
// through it, so that there is exactly one lookup in the tree and the console's
// `golden(point, object)` verb reads the same bytes the regression validation
// compares against.  It is READ-ONLY: it computes nothing, pins nothing and
// adds no object, which is why it cannot move a golden value.
//
//   point   the harness's point spelling: "n=10 ell=691" for a cohomology
//           object, "n=10 ell=691 p=2 r=1" for a Hecke one (GoldenHarness.cpp,
//           PointNE / RunCell32).
//   object  the object name: "dim_H^1", "A(T_p)^1", "Theta_flat^0", ...
//
// Returns nullptr when the pair is not in the reference.  `rows` carries the
// VALUE of a scalar invariant (`cols == 0`), the row count otherwise.
struct GoldenEntry {
    const char* point;
    const char* object;
    char tier;      // 'A' = acceptance (basis-independent), 'B' = change detector
    uint64_t rows;  // or the scalar value for scalar invariants
    uint64_t cols;  // 0 for scalar invariants
    uint64_t hash;  // FNV-1a-64, src/ModArithmetic/Fnv.h
};

const GoldenEntry* GoldenLookup(const std::string& point, const std::string& object);

// (slot-24 BUILD-SLIM, F2) The frozen reference table itself, compiled ONCE (GoldenHarness.cpp, from
// GoldenReference.inc).  Terminated by a sentinel whose `point` is nullptr; iterate `for (e = kGoldenReference; e->point; ++e)`.
// The Campaign validations read it through this declaration instead of re-including the .inc.
extern const GoldenEntry kGoldenReference[];

// (slot-24 D6) One computed golden record, as the harness's collector holds it.
struct GoldenComputedRecord {
    std::string point, object;
    char tier;
    uint64_t rows, cols, hash;
};
// The STAGED descent points (`DESCENT N=<N> n=<n>`, GoldenHarness.cpp's kGridDescent) in emission order: the
// records the ONE add-only event of slot-24 D6 will add (docs/implementation/slot24/d6golden/expect.tsv).  Until the
// owner-gated landing they are NOT part of ComputeAll() (so GoldenRegressionValidation is unchanged); the landing
// commit adds the call after the HIDAPAIR loop.  Read by IntegralDescentRecordValidation's arm (G).
std::vector<GoldenComputedRecord> GoldenDescentRecords();

// The number of entries in the compiled-in reference (14049 today) -- so a
// caller can assert the reference is present without walking it.
size_t GoldenReferenceSize();

// ===========================================================================
// (hecke-extensions slot 6, WP-6 / E2 close-out) Exposed harness recipes.
// ===========================================================================
//
// HeckeExtCampaignValidation's golden arm compares the campaign's own
// in-memory `hecke_ext::HeckeExtRecord` against GoldenReference.inc on the
// SETTINGS-INDEPENDENT objects only ([X] Sec. 1.6; owner ruling, slot-6
// Sec. 6 decision 5: `ext_json_*` equality is booked to slot 7, next to the
// harness's `source`-argument fix). It must use the harness's OWN recipes,
// never a copy, so GoldenHarness.cpp exposes them here as thin forwarders
// onto the same anonymous-namespace functions GoldenRegressionValidation's
// ext sweep calls; there is exactly one definition of each. The same
// `RedactJsonKeys` forwarder is what HeckeExtDeterminismValidation uses to
// redact the record's `commit` key (baked in at CMake configure time, so it
// is not reproducible across checkouts unredacted) before hashing.
namespace hecke_ext { struct HeckeExtRecord; class SturmBound; }

std::string GoldenNormalizeJsonWhitespace(const std::string& s);
std::string GoldenRedactJsonKeys(const std::string& normalized,
                                 const std::vector<std::string>& keys);
// (slot 7, WP-2 / [X] Sec. 8 stage 1 (1)) The path-scoped settings recipe:
// the TOP-LEVEL `settings` member of a whitespace-free record text reduced to
// `{"fingerprint":...}`, nothing else touched.  The golden ext sweep applies
// it to `ext_json_canonical` / `ext_json_full`; HeckeExtDeterminismValidation
// (owner decision 2) and HeckeExtCampaignValidation's golden arm apply the
// SAME definition through this forwarder.
std::string GoldenReduceSettingsToFingerprint(const std::string& normalized);
uint64_t GoldenHashExtIdeals(const hecke_ext::HeckeExtRecord& r);
uint64_t GoldenHashExtCertify(const hecke_ext::HeckeExtRecord& r);
uint64_t GoldenHashSturmBound(hecke_ext::SturmBound& B);

#endif // GOLDEN_HARNESS_H
