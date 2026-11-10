// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// Phase 0 -- golden-output regression harness.
//
// Plan references: docs/matrix-engine/00-implementation-plan.md ("Phase 0",
// bullet "Golden harness", step (5); Sec. 4.1; D5b; Sec. 7 Q1) and
// docs/matrix-engine/02-pipeline-consumer-audit.md Sec. 6 (bit-identity
// assertions, certificates, determinism hazards).
//
// PURPOSE. Freeze a bit-exact reference of everything the shipped pipeline
// computes BEFORE any arithmetic change, so every later phase is accepted only
// if it reproduces these values. The reference was emitted ONCE from the
// unmodified `c8227c8`-equivalent tree (branch matrix-engine tip e629344 =
// c8227c8 code + docs) plus this harness only, built with the shipped
// -O0 configuration (CMAKE_BUILD_TYPE=Debug), and checked in as
// Golden/GoldenReference.inc; it was RE-BASELINED once, on 2026-09-16, by
// owner decision, for the iota-symmetric canonical tau (every changed line
// classified object by object in
// docs/matrix-engine/rebaseline-2026-09-16-iota-symmetric-tau.md: only the
// cochain-level Theta objects and the JSON reports containing them moved).
// Release, Debug and both HECKE_PURE_INTEGER settings must reproduce it
// exactly.
//
// OWNER DECISION Q1 = "only canonical objects" => two tiers:
//
//   Tier A (acceptance, must never change) -- basis-INDEPENDENT objects:
//     per (n, ell):
//       dickson                    IsDicksonExceptional(n, ell) as 0/1 (pins the predicate)
//       dim_H^q                    dim H^q(K,R;L), q = 0,1,2
//       dim_C_eq^q, dim_Z_eq^q,
//       dim_B_eq^q                 subspace dimensions (= basis cardinalities)
//     per (n, ell, p) [point strings "... p=<p> r=1" / "... p=<p> r=2"]:
//       Theta_flat^q               (N_q d_V)^2 cochain-level matrix of T_p in the FIXED
//                                  simplicial (x) Sym basis (dims, FNV-1a-64 hash); r = 1
//       rank_Theta_flat^q
//       charpoly_A(T_p)^q, rank_A(T_p)^q, trace_A(T_p)^q          (r = 1)
//       charpoly/rank/trace of     A(T_p^2)_rec  (algebra recursion)                (r = 2)
//                                  A(T_p^2)_dir  (independent direct double-coset route)
//                                  A'(T_p^2)_prim (primitive double coset, N = p^2 pipeline)
//       T(p,p)^q                   the diamond scalar matrix p^n I (canonical as a matrix)
//       decomp_A(T_p)^q            single-operator primary decomposition invariants of
//                                  A^{(q)}(T_p): block count, per block (dim, residue
//                                  degree, char poly, min poly, Jordan partition), in the
//                                  shipped canonical block order; and certify() == 0
//       report_canonical^q         HeckeReport JSON (level (a) Theta_flat + level (b) +
//                                  level (c)) after whitespace normalisation with the
//                                  basis-dependent values REDACTED (keys "level_b",
//                                  "basis_columns", "change_of_basis")
//     per (n, ell) at q = 1:
//       decomp_sim{2,3,5}          simultaneous MaxSpec(T) decomposition invariants of the
//                                  commuting family {T_2, T_3, T_5} (block dims, residue
//                                  degrees, per-p char/min polys, Jordan partitions)
//     64-bit points (HeckeExtendedRegimeValidation's ell in {8589934627, 2^61-1},
//       n in {0,2,10}, p in {2,3}, r = 1): dim_H^q, Theta_flat^q, rank_Theta_flat^q,
//       charpoly/rank/trace of A(T_p)^q.
//     S6 induced-map cases (HeckeMorphismRegressionValidation part 1, same public-API
//       call sequences): rank_induced^q always, charpoly_induced^q where square.
//
//   Tier B (change detector; re-baselineable with a documented note) -- basis-
//   DEPENDENT hashes (D5b: all rest on the legacy kernel-basis convention):
//     C_eq^q, Z_eq^q, B_eq^q, H_eq^q bases; projectors P^q; Theta_eq^q (restriction
//     of Theta_flat to C_eq in the stored C_eq basis -- extracted from the r=1
//     HeckeOperatorOnK to_json(), see below); the raw matrices A(T_p)^q,
//     A(T_p^2)_rec^q, A(T_p^2)_dir^q, A'(T_p^2)_prim^q; the decomposition change of
//     basis U (decomp_U^q, decomp_sim_U); the residue-degree witness data of the
//     simultaneous decomposition (RNG-driven search, hazard 02 Sec. 6.5(4));
//     report_full^q = the full HeckeReport JSON hash after whitespace normalisation;
//     the S6 raw induced matrices; the 64-bit raw A(T_p)^q and bases.
//
//   How the pipeline is driven. Theta_eq^q and the HeckeOperatorOnK JSON are only
//   exposed by HeckeOperatorOnK itself, and HeckeOperatorFamily does not hand out its
//   pipeline objects. So per (n, ell, p) the r = 1 objects (Theta_flat, Theta_eq, A,
//   op_json_full) come from ONE explicitly built HeckeCosetTables -> HeckeTauTable ->
//   HeckeOperatorOnK pipeline, and the algebra objects (T_p^2 recursion / direct /
//   primitive, T(p,p), and A(T_p)_via_family) from a HeckeOperatorFamily on the same
//   cohomology object. The family re-runs the r = 1 pipeline internally for T_p (one
//   duplicate run per (n, ell, p); its A matrices are pinned as A(T_p)_via_family so
//   the two runs are also cross-checked). The r = 2 (N = p^2) Theta_flat is NOT
//   captured (no shipped accessor without a third pipeline run) -- documented gap.
//   Nothing in this file re-implements shipped mathematics; every object is read
//   from a public accessor.
//
// HASH. FNV-1a-64 (offset 0xcbf29ce484222325, prime 0x100000001b3), fed byte-wise
// (little-endian) with: rows, cols, then every entry widened to uint64_t in the L1
// row-major flat layout. For a basis (vector of matrices): count, then each matrix as
// above. For a polynomial: number of coefficients, then coefficients low -> high.
// For a scalar invariant (dimension, rank, trace, flag): the entry stores the value
// itself in `rows` (cols = 0) and the hash of that single uint64.
//
// GRID (00-implementation-plan Phase 0): n_sym in {0,1,2,4,10,16,22,24,32} x
// ell in {2,3,5,13,691,10007,134217757} x p_hecke in {2,3,5} x r in {1,2}, q in
// {0,1,2}, plus the 64-bit points -- TRIMMED at n_sym = 32 to the single cell
// (32, 5), see the TRIMMING DECISION below. "Mathematically excluded" and "Dickson-
// exceptional" are decided by calling shipped predicates: a point is excluded iff
// the shipped constructors reject it with std::invalid_argument (recorded as an
// `excluded` tier-A entry hashing the message; in this grid none does), and the
// Dickson predicate is the verbatim S12 IsDicksonExceptional (it lives in an
// anonymous namespace in HeckeEndToEndValidation.cpp, which this harness may not
// edit; the copy is pinned as the `dickson` tier-A entry so a divergence is caught).
// Dickson-exceptional points ((24,5), (32,5) here) are hashed like every other point;
// this harness asserts NO oracle values, only reproducibility.
//
// TRIMMING DECISION: see kGridNSym below and the note next to it.
//
// FILE FORMAT of GoldenReference.inc: a plain brace-initialiser list of
// GoldenEntry{point, object, tier, rows, cols, hash} lines, one per object, in
// emission order, preceded by comment lines. Hand-rolled I/O, no third-party
// libraries (repo convention). GoldenEmit writes it (only with GOLDEN_EMIT=1 set,
// see the interlock in GoldenEmit()); GoldenRegressionValidation #includes it. The emitted bytes contain NO timestamps or timings, so two emissions
// from the same binary must be byte-identical (determinism check), and emissions from
// Debug / Release / HECKE_PURE_INTEGER=ON must agree on every data line (the
// provenance comment block records the emitting build's compiler flags).

#include "GoldenHarness.h"

#include "../EquivariantCochainComplexMorphismOnK.h"
#include "../EquivariantCohomologyOnK.h"
#include "../LocalSystemPresets.h"
#include "../LocalSystemSL2Z.h"
#include "../LocalSystemSimplicialComplex.h"
#include "../LocalSystemUtils.h"
#include "../SimplicialCohomology.h"
#include "../Hecke/CoefficientRing.h"
#include "../Hecke/DicksonCount.h"
#include "../Hecke/FpPoly.h"
#include "../Hecke/HeckeCosetTables.h"
#include "../Hecke/HeckeModuleDecomposition.h"
#include "../Hecke/HeckeOrdinaryPart.h"
#include "../Hecke/HeckeOperatorFamily.h"
#include "../Hecke/HeckeOperatorOnK.h"
#include "../Hecke/HeckeReport.h"
#include "../Hecke/HeckeTauTable.h"
#include "../Hecke/HeckeBoundaryDatum.h"
#include "../Hecke/HeckeBoundaryOperator.h"
#include "../Hecke/HeckeConeAction.h"
#include "../Hecke/HeckeContragredient.h"
#include "../Hecke/HeckeLevelNDegen.h"
#include "../Hecke/HeckeLevelNSparse.h"
#include "../Hecke/HeckeNebentypus.h"
#include "../Hecke/HeckePairHomotopy.h"
#include "../Hecke/HeckePairing.h"
#include "../Hecke/HeckeReduction.h"
#include "../Hecke/HeckeSettings.h"
#include "../Hecke/SL2ZBoundaryPair.h"
#include "../Hecke/SL2ZTessellationTables.h"
#include "../EquivariantMappingCone.h"
#include "../HeckeExt/HeckeBoundary.h"
#include "../HeckeExt/HeckeCanonicalField.h"
#include "../HeckeExt/HeckeExtDriver.h"
#include "../HeckeExt/HeckeExtRecord.h"
#include "../HeckeExt/HeckeExtReport.h"
#include "../HeckeExt/HeckeHidaTower.h"
#include "../HeckeExt/HeckeRingHeckeAlgebra.h"
#include "../HeckeExt/HeckeSignSplit.h"
#include "../HeckeExt/HeckeSturmBound.h"
#include "../HeckeExt/LValue/Bernoulli.h"
#include "../HeckeExt/LValue/BirchModularSymbol.h"
#include "../HeckeExt/LValue/EisensteinBernoulliRoute.h"
#include "../HeckeExt/LValue/HidaPairing.h"
#include "../HeckeExt/LValue/LValueRoute.h"
#include "../HeckeExt/LValue/MBlockReader.h"
#include "../Integral/Descent/DescentRecord.h"
#include "../Hecke/HeckeLevelPair.h"                 // (slot 23, G-EVT23; ported slot 25 GOLD-PORT) LESLEVEL
#include "../Hecke/HeckeTameFamily.h"                // TAMEFAM
#include "../HeckeExt/HeckeLambdaTower.h"            // TOWER
#include "../HeckeExt/LValue/Gamma0NShapiro.h"       // SHAPIRO
#include "../HeckeExt/HeckeZRecordSource.h"          // (slot 25, HZ-GOLD) HZ: DefaultRecordSources()
#include "../Integral/HeckeZRecord.h"                // HZ: integral_hecke_structure/v1
#include "../NumberField/Ambiguous.h"                // PRANK
#include "../NumberField/BigInt.h"
#include "../NumberField/ClassGroupRel.h"
#include "../NumberField/Sha256.h"
#include "../../ModArithmetic/EngineCheck.h"
#include "../../ModArithmetic/FlatMatrix.h"
#include "../../ModArithmetic/Fnv.h"
#include "../../ModArithmetic/Matrix.h"
#include "../../ModArithmetic/ModArithmetic.h"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <ostream>
#include <system_error>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

using Mat32 = MatrixT<uint32_t>;
using Mat64 = MatrixT<uint64_t>;

// ===========================================================================
// Grid.
// ===========================================================================

// TRIMMING DECISION (measured 2026-09-05 on the shipped -O0 Debug build, GCC
// 11.4.0, single-threaded): the full plan grid n_sym in {0,1,2,4,10,16,22,24,32}
// took 929 s (15.5 min) for the 32-bit cells, of which n_sym = 32 alone took
// 497 s (49-94 s per ell; the r = 2, p = 5 pipeline with 30 cosets dominates).
// That exceeds the ~10 min budget, so n_sym = 32 is dropped from the regular
// grid EXCEPT for the single Dickson-exceptional cell (n, ell) = (32, 5), which
// is kept (kExtraCells) so that BOTH Dickson loci of the plan grid, (24, 5) and
// (32, 5), stay pinned. Per-n cost of the retained grid: n<=4: 6 s, n=10: 17 s,
// n=16: 57 s, n=22: 153 s, n=24: 197 s, (32,5): 55 s -> ~8 min total at -O0.
const std::vector<uint32_t> kGridNSym = {0, 1, 2, 4, 10, 16, 22, 24};
const std::vector<uint32_t> kGridEll = {2, 3, 5, 13, 691, 10007, 134217757};
const std::vector<uint64_t> kGridP = {2, 3, 5};
const std::vector<std::pair<uint32_t, uint32_t>> kExtraCells = {{32, 5}}; // Dickson locus

// HeckeExtendedRegimeValidation's 64-bit points (verbatim parameters).
const std::vector<uint64_t> kGridEll64 = {8589934627ull, 2305843009213693951ull};
const std::vector<uint32_t> kGridNSym64 = {0, 2, 10};
const std::vector<uint64_t> kGridP64 = {2, 3};

// S6 SL2Z -> S^1-cusp restriction cases (HeckeMorphismRegressionValidation
// part 1 uses (8,5), (10,5), (10,691), (100,5); n = 100 is omitted here as it
// alone would cost ~2 min at -O0 -- the S6V validation keeps pinning it).
const std::vector<std::pair<uint32_t, uint32_t>> kGridS6Restriction = {{8, 5}, {10, 5}, {10, 691}};

// (hecke-extensions slot 8, WP-4; E8 item 4a; [E] v1.9.0-pre.1 sec:ring (e);
// slot-8 plan Sec. 2.3 item 5, decision 5 (A)) THE E8 POINT LIST -- its own,
// looped by RunE8Point from ComputeAll() ONLY (after the S6 loop), never from
// RunCell32, so GoldenSubset32 (EngineDeterminismValidation) cannot reach it.
// (6, 5), (30, 5), (60, 5) are NOT grid points and do not join the grid.
// Each entry (n, ell, k) is the point `E8 ring=Z/<ell>^<k> n=<n> ell=<ell>`.
// Primes: {2, 3, 5} at (10, 691); {2, 3} at ell = 5 (there p = 5 is U_5,
// validated in slot 8 by HeckeReductionValidation but not frozen).
struct E8Point { uint32_t n; uint32_t ell; uint32_t k; std::vector<uint64_t> primes; };
const std::vector<E8Point> kGridE8 = {
    {10, 691, 2, {2, 3, 5}},
    {6, 5, 2, {2, 3}},
    {16, 5, 2, {2, 3}},
    {30, 5, 2, {2, 3}},
    {30, 5, 3, {2, 3}},
    {60, 5, 2, {2, 3}},
    {60, 5, 3, {2, 3}},
};

// ===========================================================================
// The reference entry type (shared with GoldenReference.inc).
//
// (console phase C1a, inventory gap G13) The struct was file-local until the
// console needed to READ the reference from a verb.  It now lives in
// GoldenHarness.h, verbatim, together with the one public lookup GoldenLookup;
// nothing else about it changed, so every entry of GoldenReference.inc still
// initialises exactly the same aggregate and no golden value can move.
// ===========================================================================

} // namespace (reopened below): the table has external linkage, declared in GoldenHarness.h

const GoldenEntry kGoldenReference[] = {
#include "GoldenReference.inc"
    // Sentinel so the array is never empty (a zero-length array is ill-formed).
    {nullptr, nullptr, 'A', 0, 0, 0}
};

namespace {

// Provenance note written into the emitted file (constant by design: the
// emitted bytes must not depend on run time or run order).
const char* const kProvenanceNote =
    "Emitted by `./cohomology_app GoldenEmit`. Baseline history: first emitted from\n"
    "// the unmodified c8227c8-equivalent pipeline (branch matrix-engine tip e629344)\n"
    "// at -O0; RE-BASELINED 2026-09-16 (owner decision) for the iota-symmetric\n"
    "// canonical tau of [C, alg:tau] -- iota_K-equivariant tie-break in step (1b)\n"
    "// and tree-routed path templates in step 2 -- which moved every Theta_flat /\n"
    "// rank_Theta_flat / Theta_eq / report / op_json entry (2097 lines) and NO\n"
    "// dimension, A^{(q)}, decomposition, basis or projector entry; see\n"
    "// docs/matrix-engine/rebaseline-2026-09-16-iota-symmetric-tau.md. Debug,\n"
    "// Release, Bench and HECKE_PURE_INTEGER=ON/OFF must reproduce every data line\n"
    "// below byte-for-byte.\n"
    "// RE-BASELINED 2026-09-16 (owner decision, hecke-extensions slot 2, phase\n"
    "// E0b) for hecke_report/v2: exactly the 384 report_canonical^q /\n"
    "// report_full^q lines, 0 elsewhere, 0 splits; see\n"
    "// docs/matrix-engine/rebaseline-2026-09-16-hecke-report-v2.md.\n"
    "// EXTENDED 2026-09-22 (owner decision, hecke-extensions slot 5, WP-3) by the\n"
    "// extension stage's ONE ADD-ONLY emission: 2617 NEW entries (1468 tier A,\n"
    "// 1149 tier B) for the E2 / E3 / E4b / E5 objects of [X] Sec. 1.6 on the ext\n"
    "// subgrid, and ext_excluded off it.  golden_explain.py reported 0 changed and\n"
    "// 0 removed lines and 236 unchanged decompositions; not one pre-existing data\n"
    "// line moved.  See docs/matrix-engine/rebaseline-2026-09-22-hecke-ext-v2.md.\n"
    "// EXTENDED 2026-09-23 (hecke-extensions slot 7, WP-1, stage 0; predicate-gated\n"
    "// by owner ruling 2 (B)) by the SEPARATE E7 add-only emission: 651 NEW entries\n"
    "// (558 tier A, 93 tier B) under the point suffix ` e7` on the 31 ext-subgrid\n"
    "// cells.  golden_ref_diff.py: added = the pre-stated set, changed = removed =\n"
    "// empty.  See docs/matrix-engine/rebaseline-2026-09-23-e7-addonly.md.\n"
    "// RE-BASELINED 2026-09-23 (hecke-extensions slot 7, WP-2, stage 1 (1);\n"
    "// predicate-gated by owner ruling 2 (B)): the path-scoped settings recipe --\n"
    "// the top-level `settings` of the ext record reduced to its fingerprint --\n"
    "// changed exactly the 62 ext_json_canonical / ext_json_full lines, added and\n"
    "// removed nothing (golden_ref_diff.py --expect); golden_explain.py: only the\n"
    "// eight settings.<key> paths.  From here a key registered at its default\n"
    "// moves no golden line.  See\n"
    "// docs/matrix-engine/rebaseline-2026-09-23-hecke-ext-staged.md.\n"
    "// RE-BASELINED 2026-09-23 (slot 7, WP-3, stage 2 item 3; owner-signed): the\n"
    "// cell-local proof_range -- 64 changed (the 62 ext_json_* lines + ext_ideals /\n"
    "// ext_certify at n=24 ell=5), nothing added or removed; same note.\n"
    "// RE-BASELINED 2026-09-23 (slot 7, WP-3, stage 2 item 4; owner-signed): S_sep\n"
    "// in the default Stage 2(3) -- the 62 ext_json_* lines (the `s_sep` object;\n"
    "// T_7 at ell = 13), no raw object; same note.\n"
    "// RE-BASELINED 2026-09-23 (slot 7, WP-3, stage 2 item 5; owner-signed): the\n"
    "// per-ideal atlas `label` -- the 62 ext_json_* lines, no raw object; same note.\n"
    "// RE-BASELINED 2026-09-23 (slot 7, WP-3, stage 2 item 6; owner-signed): the\n"
    "// canonical split `split` as the default report form -- the 62 ext_json_*\n"
    "// lines, no raw object; same note.\n"
    "// RE-BASELINED 2026-09-23 (slot 7, WP-3, stage 2 item 7; owner-signed):\n"
    "// predictions_mode, predictions and the per-ideal predicted / confirmed /\n"
    "// unpredicted -- the 62 ext_json_* lines, no raw object; same note.\n"
    "// RE-BASELINED 2026-09-23 (slot 7, WP-3, stage 3; predicate-gated): the\n"
    "// schema string hecke_ext/v2 -> hecke_ext/v3 -- the 62 ext_json_* lines,\n"
    "// path `schema` only; same note.\n"
    "// EXTENDED 2026-09-24 (hecke-extensions slot 8, WP-4, E8 item 4a;\n"
    "// predicate-gated by owner ruling 5 (A)) by the E8 add-only emission: 124 NEW\n"
    "// entries (103 tier A, 21 tier B) at the seven NEW points `E8 ring=Z/<ell>^<k>\n"
    "// n=<n> ell=<ell>`, nothing elsewhere.  golden_ref_diff.py: added = the\n"
    "// pre-stated set, changed = removed = empty.  See\n"
    "// docs/matrix-engine/rebaseline-2026-09-24-e8-addonly.md.\n"
    "// EXTENDED (hecke-extensions slot 9, WP-4, E8 item 4b; R1, predicate-gated by\n"
    "// owner ruling 3 (A)) by the 4b add-only emission: 126 NEW entries (91 tier A,\n"
    "// 35 tier B) -- the boundary and cone objects of [E] sec:ring (f) -- at the\n"
    "// SEVEN EXISTING E8 points, after their E8 item 4a objects; nothing elsewhere.\n"
    "// golden_ref_diff.py: added = the pre-stated set, changed = removed = empty.\n"
    "// See docs/matrix-engine/rebaseline-2026-09-24-e8-4b-addonly.md.\n"
    "// RE-BASELINED 2026-09-26 (hecke-extensions slot 12, S12-FIN; owner-approved\n"
    "// 2026-09-26; predicate-gated): the finality certificates (d0) \"dickson\" /\n"
    "// (d1) \"boundary\" and the schema hecke_ext/v3 -> hecke_ext/v4 -- the 62\n"
    "// ext_json_* lines (schema, certificate_routes.d0 / .d1, boundary_reps), no\n"
    "// raw object: under u_ell = off no (d0) / (d1) can pass, so no final moved.\n"
    "// See docs/matrix-engine/rebaseline-2026-09-26-hecke-ext-v4-finality.md.\n"
    "// EXTENDED 2026-09-26 (hecke-extensions slot 13, WP-5; owner Q5 = yes;\n"
    "// predicate-gated) by the first Hida add-only emission: 21 NEW entries (21 tier\n"
    "// A, 0 tier B) -- the slot-12 ledger's candidate list (the pre-pass e_max and\n"
    "// torsion multiset, the ordinary ranks on H_{e'}, chi^ord(T_2), chi^ord(T_3) mod\n"
    "// ell^2) -- at nine NEW points `HIDA n=<n> ell=<ell>`, nothing elsewhere.\n"
    "// golden_ref_diff.py: added = the pre-stated set, changed = removed = empty.\n"
    "// See docs/matrix-engine/rebaseline-2026-09-26-hida-addonly.md.\n"
    "// RE-BASELINED 2026-09-26 (hecke-extensions slot 13, WP-7; owner Q3 = yes;\n"
    "// predicate-gated): (d1)'s data in the record, the schema hecke_ext/v4 ->\n"
    "// hecke_ext/v5 -- the 62 ext_json_* lines (the schema string only), no raw\n"
    "// object: under u_ell = off no (d1) passes, so no boundary_data is emitted.\n"
    "// See docs/matrix-engine/rebaseline-2026-09-26-hecke-ext-v5-d1-data.md.\n"
    "// EXTENDED 2026-10-02 (hecke-extensions slot 17, WP-4; owner O2 = yes;\n"
    "// predicate-gated) by the batched add-only emission of S16-O3: NEW points\n"
    "// `LEVELN N=<N> n=<n> p=1000003` (level N, kappa_N, alpha_d), `LOEWY_RING\n"
    "// n=<n> ell=<ell> kmax=<K>` (certified LOEWY_RING lines) and `NEB N=<N>\n"
    "// chi=<N.m> k=<k> ell=<ell>` (nebentypus), nothing elsewhere.\n"
    "// golden_ref_diff.py: added = the pre-stated set, changed = removed = empty.\n"
    "// See docs/matrix-engine/rebaseline-2026-10-02-addonly-levelN-loewy-neb.md.\n"
    "// EXTENDED 2026-10-03 (hecke-extensions slot 18, WP-4; owner O2 = yes;\n"
    "// predicate-gated) by the add-only emission (slot-18 R1): NEW points `n=<n> ell=<ell>\n"
    "// ext tau` (tau / tau_bd table hashes at (p, nu) in {(3,1), (5,1), (7,1),\n"
    "// (2,2), (3,2)}, p != ell, on the 31 ext-subgrid cells) and the d = 2 point\n"
    "// `NEB N=7 chi=7.2 k=4 ell=5` (over F_25), nothing elsewhere.\n"
    "// golden_ref_diff.py: added = the pre-stated set, changed = removed = empty.\n"
    "// See docs/matrix-engine/rebaseline-2026-10-03-addonly-tau-neb2.md.\n"
    "// EXTENDED 2026-10-09 (hecke-extensions slot 22, G-EVT; owner-approved slot-22 plan\n"
    "// decision, predicate-gated) by the add-only emission G1 / G2 / G4 / G7 of\n"
    "// the slot-20 candidate list: 59 NEW tier-A entries at the NEW points\n"
    "// `LVALUE n=<n> ell=<ell>` (v(#Q_m), the cross-kind depth, c_f^eps) and\n"
    "// `HIDAPAIR k=<k> ell=<ell>` (the 27 Hida pairing valuation tables), nothing\n"
    "// elsewhere.  golden_ref_diff.py: added = the pre-stated set, changed =\n"
    "// removed = empty.  See docs/implementation/slot22/golden_ledger.md and\n"
    "// docs/matrix-engine/rebaseline-2026-10-09-addonly-lvalue-hidapair.md.\n"
    "// EXTENDED (hecke-extensions slot 24, DESCENT D6, the ONE add-only event of\n"
    "// the integral descent; [X] OD-9, owner ruling R-Q6 of 2026-10-24,\n"
    "// predicate-gated) by 156 NEW tier-A entries at the NEW points\n"
    "// `DESCENT N=<N> n=<n>` (nine cells; the record integral_descent_record/v1\n"
    "// under coefficient_ring = Z: H0..H3, bad_primes, charZ_T<p>,\n"
    "// tors_q<q>_ell<ell>, pair_H / pair_Hd / pair_Hc), nothing elsewhere.\n"
    "// golden_ref_diff.py: added = the pre-stated set, changed = removed =\n"
    "// empty.  See docs/implementation/slot24/d6golden/ledger.md and\n"
    "// docs/matrix-engine/rebaseline-s24-d6-addonly-descent.md (they record the\n"
    "// landing's date: this text is final before the emission).\n"
    "// EXTENDED (hecke-extensions slot 25, HZ-GOLD, the ONE add-only event of\n"
    "// slot 25; owner go of 2026-11-07, predicate-gated) by 112 NEW tier-A\n"
    "// entries at NEW points, nothing elsewhere: the 97 slot-23 G-EVT23 entries\n"
    "// (ported by GOLD-PORT) at `TAMEFAM p=5 l0=11 l1=<l1>`, `PRANK N=<N> p=5`,\n"
    "// `PRANK chevalley p<=97`, `TOWER ell=<l> a=<a> r=<r>`, `SHAPIRO N=<N> n=<n>\n"
    "// ell=<l>` and `LESLEVEL`, and 15 HeckeZ entries at `HZ N=1 n=<n> alg=full`\n"
    "// (n = 2, 4, ..., 30; object HZ_N1_n<n>_full = the record_fnv1a64 of the\n"
    "// default integral_hecke_structure/v1 record, overall status final).\n"
    "// golden_ref_diff.py: added = the pre-stated set, changed = removed =\n"
    "// empty.  See docs/implementation/slot25/golden_ledger.md and\n"
    "// docs/matrix-engine/rebaseline-s25-hzgold-addonly-gevt23-heckez.md (they\n"
    "// record the landing's date: this text is final before the emission).";

// ===========================================================================
// FNV-1a-64 -- unified (hecke-extensions WP-2, [X Sec.1.1]): the offset,
// prime and byte-wise mixing live in src/ModArithmetic/Fnv.h,
// which Campaign/CampaignDriver.cpp and PerformanceSmokeValidation.cpp now
// also use, so a value that lies on both the golden grid and a campaign
// point reproduces the identical hash by construction, not by copy-paste
// agreement. `kFnvOffset` is kept as a local alias so every call site below
// (`HashMatrixInto`, `HashScalar`, the collectors) is unchanged.
// ===========================================================================

constexpr uint64_t kFnvOffset = engine::kFnvOffset64;
using engine::FnvU64;
using engine::FnvBytes;

template<class Elt>
uint64_t HashMatrixInto(uint64_t h, const MatrixT<Elt>& M) {
    h = FnvU64(h, M.rows);
    h = FnvU64(h, M.cols);
    for (uint32_t r = 0; r < M.rows; ++r)
        for (uint32_t c = 0; c < M.cols; ++c)
            h = FnvU64(h, static_cast<uint64_t>(M.data[r][c]));
    return h;
}

template<class Elt>
uint64_t HashFlatInto(uint64_t h, const FlatMatrixT<Elt>& M) {
    h = FnvU64(h, M.get_rows());
    h = FnvU64(h, M.get_cols());
    for (uint32_t r = 0; r < M.get_rows(); ++r)
        for (uint32_t c = 0; c < M.get_cols(); ++c)
            h = FnvU64(h, static_cast<uint64_t>(M.get_entry(r, c)));
    return h;
}

uint64_t HashScalar(uint64_t v) { return FnvU64(kFnvOffset, v); }

template<class Elt>
uint64_t HashPolyInto(uint64_t h, const hecke::FpPolyT<Elt>& P) {
    h = FnvU64(h, P.coeffs.size());
    for (Elt c : P.coeffs) h = FnvU64(h, static_cast<uint64_t>(c));
    return h;
}

// ===========================================================================
// (hecke-extensions phase E0a) The two optional observers.
//
// Both are NULL on every ordinary golden run (GoldenRegressionValidation,
// GoldenEmit, GoldenSubset32), so neither can move a hash: they are only
// installed by GoldenDump() and GoldenLocalitySweep(), which compute exactly
// what ComputeAll() computes and then read the finished records.
// ===========================================================================

struct DumpSink;
DumpSink* g_dump = nullptr;
GoldenLocalitySweepResult* g_sweep = nullptr;

void DumpText(const std::string& point, const std::string& object, const std::string& payload);

// ===========================================================================
// Record collector.
// ===========================================================================

struct Record {
    std::string point;
    std::string object;
    char tier;
    uint64_t rows, cols, hash;
};

class Collector {
public:
    void scalar(const std::string& point, const std::string& obj, char tier, uint64_t v) {
        recs_.push_back(Record{point, obj, tier, v, 0, HashScalar(v)});
    }
    template<class Elt>
    void matrix(const std::string& point, const std::string& obj, char tier, const MatrixT<Elt>& M) {
        recs_.push_back(Record{point, obj, tier, M.rows, M.cols, HashMatrixInto(kFnvOffset, M)});
    }
    template<class Elt>
    void flat(const std::string& point, const std::string& obj, char tier, const FlatMatrixT<Elt>& M) {
        recs_.push_back(Record{point, obj, tier, M.get_rows(), M.get_cols(), HashFlatInto(kFnvOffset, M)});
    }
    template<class Elt>
    void basis(const std::string& point, const std::string& obj, char tier,
               const std::vector<MatrixT<Elt>>& B) {
        uint64_t h = FnvU64(kFnvOffset, B.size());
        uint64_t rows = 0, cols = 0;
        for (const auto& M : B) { h = HashMatrixInto(h, M); rows = M.rows; cols = M.cols; }
        // dims: (count, rows*cols of one element) -- the per-element shape is uniform.
        recs_.push_back(Record{point, obj, tier, B.size(), rows * cols, h});
    }
    // (matrix-engine Phase 3) FlatMatrixT overloads. HashFlatInto feeds the FNV
    // state exactly the same bytes as HashMatrixInto (rows, cols, then every
    // entry in row-major order), so a record whose object moved from MatrixT to
    // FlatMatrixT keeps its frozen golden hash.
    template<class Elt>
    void matrix(const std::string& point, const std::string& obj, char tier, const FlatMatrixT<Elt>& M) {
        recs_.push_back(Record{point, obj, tier, M.get_rows(), M.get_cols(), HashFlatInto(kFnvOffset, M)});
    }
    template<class Elt>
    void basis(const std::string& point, const std::string& obj, char tier,
               const std::vector<FlatMatrixT<Elt>>& B) {
        uint64_t h = FnvU64(kFnvOffset, B.size());
        uint64_t rows = 0, cols = 0;
        for (const auto& M : B) { h = HashFlatInto(h, M); rows = M.get_rows(); cols = M.get_cols(); }
        recs_.push_back(Record{point, obj, tier, B.size(), rows * cols, h});
    }
    template<class Elt>
    void poly(const std::string& point, const std::string& obj, char tier, const hecke::FpPolyT<Elt>& P) {
        recs_.push_back(Record{point, obj, tier, P.coeffs.size(), 1, HashPolyInto(kFnvOffset, P)});
    }
    void text(const std::string& point, const std::string& obj, char tier, const std::string& s) {
        recs_.push_back(Record{point, obj, tier, s.size(), 1, FnvBytes(kFnvOffset, s)});
        if (g_dump) DumpText(point, obj, s);  // (E0a) no-op unless GoldenDump installed it
    }
    void raw(const std::string& point, const std::string& obj, char tier, uint64_t rows, uint64_t cols,
             uint64_t h) {
        recs_.push_back(Record{point, obj, tier, rows, cols, h});
    }
    const std::vector<Record>& records() const { return recs_; }

private:
    std::vector<Record> recs_;
};

// ===========================================================================
// Small math helpers (all via ModArithmetic; nothing re-implemented).
// ===========================================================================

template<class Elt> Elt ModAdd(Elt a, Elt b, Elt p);
template<> uint32_t ModAdd<uint32_t>(uint32_t a, uint32_t b, uint32_t p) { return ModArithmetic::mod_add(a, b, p); }
template<> uint64_t ModAdd<uint64_t>(uint64_t a, uint64_t b, uint64_t p) { return ModArithmetic::mod_add64(a, b, p); }

template<class Elt>
uint64_t Trace(const MatrixT<Elt>& A) {
    Elt t = 0;
    for (uint32_t i = 0; i < A.rows && i < A.cols; ++i) t = ModAdd<Elt>(t, A.data[i][i], A.p);
    return static_cast<uint64_t>(t);
}

template<class Elt>
uint64_t Rank(const MatrixT<Elt>& A) {
    if (A.rows == 0 || A.cols == 0) return 0;
    MatrixT<Elt> c = A;
    return c.rank();
}

template<class Elt>
hecke::FpPolyT<Elt> CharPoly(const MatrixT<Elt>& A) {
    if (A.rows == 0) return hecke::FpPolyT<Elt>(A.p, std::vector<Elt>{Elt(1)}); // det of 0x0 = 1
    return hecke::CharPolyBerkowitz(A);
}

// Previously a verbatim copy of Stage S12's IsDicksonExceptional
// (HeckeEndToEndValidation.cpp, anonymous namespace -- not exported; this
// harness may not edit that file). (hecke-extensions WP-2) Both that copy
// and this one are now the SAME function, `hecke::IsDicksonExceptionalCount`
// (Hecke/DicksonCount.h) -- so the `dickson` tier-A entry still pins the
// predicate, but divergence is now impossible by construction rather than
// merely caught.
bool IsDicksonExceptional(uint32_t n, uint32_t ell) {
    return hecke::IsDicksonExceptionalCount(n, ell);
}

std::string Q(uint32_t q) { return "^" + std::to_string(q); }

// ===========================================================================
// JSON helpers: whitespace normalisation and tier-B key redaction.
// (Hand-rolled scanner for the repo's own hand-rolled writers.)
// ===========================================================================

std::string NormalizeJsonWhitespace(const std::string& s) {
    std::string out;
    out.reserve(s.size());
    bool in_str = false;
    for (size_t i = 0; i < s.size(); ++i) {
        const char c = s[i];
        if (in_str) {
            out.push_back(c);
            if (c == '\\' && i + 1 < s.size()) { out.push_back(s[++i]); continue; }
            if (c == '"') in_str = false;
            continue;
        }
        if (c == '"') { in_str = true; out.push_back(c); continue; }
        if (c == ' ' || c == '\t' || c == '\n' || c == '\r') continue;
        out.push_back(c);
    }
    return out;
}

// Returns the index one past the JSON value starting at s[i] (whitespace-free input).
size_t SkipJsonValue(const std::string& s, size_t i) {
    if (i >= s.size()) return i;
    if (s[i] == '"') {
        for (++i; i < s.size(); ++i) {
            if (s[i] == '\\') { ++i; continue; }
            if (s[i] == '"') return i + 1;
        }
        return s.size();
    }
    if (s[i] == '{' || s[i] == '[') {
        int depth = 0;
        bool in_str = false;
        for (; i < s.size(); ++i) {
            const char c = s[i];
            if (in_str) { if (c == '\\') ++i; else if (c == '"') in_str = false; continue; }
            if (c == '"') in_str = true;
            else if (c == '{' || c == '[') ++depth;
            else if (c == '}' || c == ']') { if (--depth == 0) return i + 1; }
        }
        return s.size();
    }
    while (i < s.size() && s[i] != ',' && s[i] != '}' && s[i] != ']') ++i;
    return i;
}

// Replaces the VALUE of every occurrence of `"key":` (any nesting) by "<redacted>".
std::string RedactJsonKeys(const std::string& normalized, const std::vector<std::string>& keys) {
    std::string s = normalized;
    for (const std::string& key : keys) {
        const std::string needle = "\"" + key + "\":";
        size_t pos = 0;
        while ((pos = s.find(needle, pos)) != std::string::npos) {
            const size_t vstart = pos + needle.size();
            const size_t vend = SkipJsonValue(s, vstart);
            s.replace(vstart, vend - vstart, "\"<redacted>\"");
            pos = vstart + 12;
        }
    }
    return s;
}

// (hecke-extensions slot 7, WP-2 / [X] Sec. 8 stage 1 (1)) The members of the
// whitespace-free JSON OBJECT starting at s[begin] (s[begin] == '{'), in
// order: {key, value begin, value end}.  Depth-1 only -- nested objects are
// skipped whole by SkipJsonValue, never searched.  ENGINE_CHECKs the shape.
struct JsonMember { std::string key; size_t vbegin; size_t vend; };
std::vector<JsonMember> JsonObjectMembers(const std::string& s, size_t begin, size_t end) {
    ENGINE_CHECK_MSG(begin < end && end <= s.size() && s[begin] == '{' && s[end - 1] == '}',
                     "JsonObjectMembers: not a JSON object");
    std::vector<JsonMember> out;
    size_t i = begin + 1;
    if (s[i] != '}') {
        while (true) {
            ENGINE_CHECK_MSG(i < end && s[i] == '"', "JsonObjectMembers: expected a key");
            const size_t kend = SkipJsonValue(s, i);
            ENGINE_CHECK_MSG(kend < end && s[kend] == ':', "JsonObjectMembers: expected ':'");
            JsonMember m;
            m.key = s.substr(i + 1, kend - i - 2);
            m.vbegin = kend + 1;
            m.vend = SkipJsonValue(s, m.vbegin);
            ENGINE_CHECK_MSG(m.vend < end, "JsonObjectMembers: truncated value");
            out.push_back(m);
            if (s[m.vend] == ',') { i = m.vend + 1; continue; }
            ENGINE_CHECK_MSG(s[m.vend] == '}' && m.vend == end - 1,
                             "JsonObjectMembers: expected ',' or the closing '}'");
            break;
        }
    }
    return out;
}

// THE PATH-SCOPED SETTINGS RECIPE ([X] Sec. 8 stage 1 (1); slot-7 plan WP-2
// scope 1).  Reduces the value of the TOP-LEVEL member `settings` of a
// whitespace-free JSON object to `{"fingerprint":<its fingerprint value>}`
// and touches NOTHING at any other path -- in particular not the top-level
// `u_ell` verdict, which `RedactJsonKeys({"u_ell"})` (any nesting) would hit.
//
// Why: the `settings` block lists every REGISTERED key with its value and
// source, so every key registration moved every golden ext text (62 lines)
// although no computed value changed.  The fingerprint alone is a function of
// the NON-DEFAULT entries only (0 at the defaults), so after this reduction a
// key registered at its default moves nothing: registration is golden-
// additive once and permanently.  A setting that IS in force still moves the
// text, through the fingerprint ([X] Sec. 1.5 rule 1).
//
// Exactly one top-level `settings` member holding exactly one depth-1
// `fingerprint` member is required (ENGINE_CHECK): a record without them is
// not a `hecke_ext` record, and silently hashing it unreduced is the defect.
std::string ReduceSettingsToFingerprint(const std::string& normalized) {
    const std::vector<JsonMember> top = JsonObjectMembers(normalized, 0, normalized.size());
    const JsonMember* settings = nullptr;
    for (const JsonMember& m : top) {
        if (m.key != "settings") continue;
        ENGINE_CHECK_MSG(settings == nullptr,
                         "ReduceSettingsToFingerprint: more than one top-level `settings`");
        settings = &m;
    }
    ENGINE_CHECK_MSG(settings != nullptr,
                     "ReduceSettingsToFingerprint: no top-level `settings` member");
    const std::vector<JsonMember> inner =
        JsonObjectMembers(normalized, settings->vbegin, settings->vend);
    const JsonMember* fp = nullptr;
    for (const JsonMember& m : inner) {
        if (m.key != "fingerprint") continue;
        ENGINE_CHECK_MSG(fp == nullptr,
                         "ReduceSettingsToFingerprint: more than one `settings.fingerprint`");
        fp = &m;
    }
    ENGINE_CHECK_MSG(fp != nullptr, "ReduceSettingsToFingerprint: no `settings.fingerprint`");
    const std::string reduced =
        "{\"fingerprint\":" + normalized.substr(fp->vbegin, fp->vend - fp->vbegin) + "}";
    std::string s = normalized;
    s.replace(settings->vbegin, settings->vend - settings->vbegin, reduced);
    return s;
}

// ===========================================================================
// Object extraction.
// ===========================================================================

std::vector<LocalSystemSimplicialComplex::SimplexRelation> BuildRelations(
        const SimplicialComplex& K, uint32_t n_sym, uint32_t ell) {
    std::vector<LocalSystemSimplicialComplex::SimplexRelation> rels;
    for (const auto& datum : GenerateSL2ZRelationsWithGroupData(K, n_sym, ell)) rels.push_back(datum.rel);
    return rels;
}

template<class Elt>
void CollectCohomology(Collector& C, const std::string& pt, const EquivariantCohomologyOnKT<Elt>& coh) {
    const auto& dims = coh.get_H_eq_dimensions();
    const auto& Ceq = coh.get_C_eq_bases_fm();
    const auto& Zeq = coh.get_Z_eq_bases_fm();
    const auto& Beq = coh.get_B_eq_bases_fm();
    const auto& Heq = coh.get_H_eq_bases_fm();
    const auto& P = coh.get_H_eq_projectors_from_C_eq_flat_fm();
    C.scalar(pt, "num_degrees", 'A', dims.size());
    for (uint32_t q = 0; q < dims.size(); ++q) {
        C.scalar(pt, "dim_H" + Q(q), 'A', dims[q]);
        C.scalar(pt, "dim_C_eq" + Q(q), 'A', q < Ceq.size() ? Ceq[q].size() : 0);
        C.scalar(pt, "dim_Z_eq" + Q(q), 'A', q < Zeq.size() ? Zeq[q].size() : 0);
        C.scalar(pt, "dim_B_eq" + Q(q), 'A', q < Beq.size() ? Beq[q].size() : 0);
        if (q < Ceq.size()) C.basis(pt, "C_eq" + Q(q), 'B', Ceq[q]);
        if (q < Zeq.size()) C.basis(pt, "Z_eq" + Q(q), 'B', Zeq[q]);
        if (q < Beq.size()) C.basis(pt, "B_eq" + Q(q), 'B', Beq[q]);
        if (q < Heq.size()) C.basis(pt, "H_eq" + Q(q), 'B', Heq[q]);
        if (q < P.size()) C.matrix(pt, "P" + Q(q), 'B', P[q]);
    }
}

// Char poly / rank / trace (tier A) + raw matrix (tier B) of a cohomology-level operator.
template<class Elt>
void CollectOperatorMatrices(Collector& C, const std::string& pt, const std::string& name,
                             const std::vector<MatrixT<Elt>>& A, char raw_tier = 'B') {
    for (uint32_t q = 0; q < A.size(); ++q) {
        C.poly(pt, "charpoly_" + name + Q(q), 'A', CharPoly(A[q]));
        C.scalar(pt, "rank_" + name + Q(q), 'A', Rank(A[q]));
        C.scalar(pt, "trace_" + name + Q(q), 'A', Trace(A[q]));
        C.matrix(pt, name + Q(q), raw_tier, A[q]);
    }
}

// ===========================================================================
// (hecke-extensions phase E0a) Dump sink and locality sweep.
// ===========================================================================

std::string Hex64(uint64_t v);  // defined with the reference I/O below

struct DumpSink {
    std::ofstream index, texts, decomps;
    uint64_t n_texts = 0, n_decomps = 0;
};

// One line per text object: point <TAB> object <TAB> payload.  Every payload is
// whitespace-normalised JSON, hence newline-free, so one record = one line.
void DumpText(const std::string& point, const std::string& object, const std::string& payload) {
    DumpSink& S = *g_dump;
    S.texts << point << "\t" << object << "\t" << payload << "\n";
    S.index << "TEXT\t" << point << "\t" << object << "\n";
    ++S.n_texts;
}

void DumpPolyInto(std::ostream& os, const hecke::FpPoly& P) {
    for (size_t i = 0; i < P.coeffs.size(); ++i) os << (i ? "," : "") << P.coeffs[i];
    if (P.coeffs.empty()) os << "-";
}

// The four golden hashes of a decomposition are functions of exactly the data
// written here, so a diff of two dumps explains -- or refutes -- any movement
// of _invariants / _certify / _witness / _U.
void DumpDecomposition(const std::string& pt, const std::string& name,
                       const hecke::HeckeModuleDecomposition& D) {
    DumpSink& S = *g_dump;
    std::ostream& os = S.decomps;
    os << "## " << pt << "\t" << name << "\n";
    os << "blocks\t" << D.blocks().size() << "\n";
    os << "certify\t" << D.certify() << "\n";
    for (size_t i = 0; i < D.blocks().size(); ++i) {
        const hecke::PrimaryBlock& b = D.blocks()[i];
        const std::string pre = "block\t" + std::to_string(i) + "\t";
        os << pre << "dim\t" << b.dim << "\n";
        os << pre << "residue_degree\t" << b.residue_degree << "\n";
        os << pre << "residue_degree_certified\t" << (b.residue_degree_certified ? 1 : 0) << "\n";
        os << pre << "residue_witness";
        for (const auto& w : b.residue_witness) os << "\t" << w.first << ":" << w.second;
        os << "\n";
        for (const auto& kv : b.char_poly) {
            os << pre << "char_poly\t" << kv.first << "\t";
            DumpPolyInto(os, kv.second);
            os << "\n";
        }
        for (const auto& kv : b.min_poly) {
            os << pre << "min_poly\t" << kv.first << "\t";
            DumpPolyInto(os, kv.second);
            os << "\n";
        }
        for (const auto& kv : b.jordan_partition) {
            os << pre << "jordan_partition\t" << kv.first << "\t";
            for (size_t j = 0; j < kv.second.size(); ++j) os << (j ? "," : "") << kv.second[j];
            if (kv.second.empty()) os << "-";
            os << "\n";
        }
        // The change-of-basis columns of this block (U is their concatenation).
        os << pre << "basis_columns\t" << b.basis_columns.rows << "x" << b.basis_columns.cols;
        for (uint32_t r = 0; r < b.basis_columns.rows; ++r)
            for (uint32_t c = 0; c < b.basis_columns.cols; ++c)
                os << "\t" << b.basis_columns.data[r][c];
        os << "\n";
    }
    S.index << "DECOMP\t" << pt << "\t" << name << "\n";
    ++S.n_decomps;
}

// Read-only lookup into the compiled-in reference (never a copied constant).
//
// (console phase C1a, [X] Sec. 1.1 "one lookup, not two") The BODY moved to the
// public ::GoldenLookup at the foot of this file; this name is kept as the
// in-file spelling the sweep already used, and is now a pure forwarder.  There
// is exactly one index over GoldenReference.inc in the tree.
const GoldenEntry* FindReference(const std::string& point, const std::string& object) {
    return ::GoldenLookup(point, object);
}

// Certifies every block of D and re-checks the FOUR records CollectDecomposition
// has just pushed (they are the last four of `recs`).
void SweepDecomposition(const std::vector<Record>& recs, const std::string& pt,
                        const std::string& name, const hecke::HeckeModuleDecomposition& D) {
    GoldenLocalitySweepResult& R = *g_sweep;
    ++R.decompositions;
    for (size_t i = 0; i < D.blocks().size(); ++i) {
        const hecke::PrimaryBlock& b = D.blocks()[i];
        ++R.blocks;
        const hecke::LocalityVerdict v = hecke::CertifyLocality(b, D.ops(), D.ell());
        if (v.analytic) ++R.analytic_blocks;
        const std::string where = pt + " / " + name + " block " + std::to_string(i) +
                                  " (dim " + std::to_string(b.dim) + ", f " +
                                  std::to_string(b.residue_degree) + ")";
        if (!v.local) {
            ++R.nonlocal_blocks;
            R.findings.push_back("NONLOCAL " + where + ": code " + std::to_string(v.failure) +
                                 " -- " + v.note);
        } else if (v.witness != b.residue_witness) {
            // Local, but the first witness that carries a FULL certificate is
            // not the one the shipped search accepted: E0b would take a
            // different branch here, so it is a finding too.
            ++R.witness_changes;
            R.findings.push_back("WITNESS " + where +
                                 ": the certified witness differs from the shipped one");
        }
        // (E0b, slot 2 WP-1) compute() now certifies locality ITSELF and
        // stores the certificate on the block.  Assert it agrees, field for
        // field, with the independent read-only verdict.
        if (b.phase_reached != hecke::LocalityPhase::Linear) ++R.nonlinear_phase_blocks;
        const bool agrees =
            (b.local_certified == v.local) &&
            (!v.local || (b.witness_poly == v.witness_poly &&
                          b.w_hat_exponent == v.w_hat_exponent &&
                          b.character == v.character));
        if (!agrees) {
            ++R.loop_disagreements;
            R.findings.push_back("LOOP " + where +
                                 ": compute()'s own locality certificate differs from the "
                                 "read-only verdict");
        }
    }
    const size_t n = recs.size();
    for (size_t k = (n >= 4 ? n - 4 : 0); k < n; ++k) {
        const Record& r = recs[k];
        ++R.hashes_checked;
        const GoldenEntry* e = FindReference(r.point, r.object);
        if (e == nullptr) {
            ++R.hashes_failing;
            R.hash_failures.push_back("MISSING " + r.point + " / " + r.object);
            continue;
        }
        if (!(e->rows == r.rows && e->cols == r.cols && e->hash == r.hash && e->tier == r.tier)) {
            ++R.hashes_failing;
            R.hash_failures.push_back("MISMATCH [tier " + std::string(1, e->tier) + "] " + r.point +
                                      " / " + r.object + ": expected " + Hex64(e->hash) + " got " +
                                      Hex64(r.hash));
        }
    }
}

// Basis-independent invariants of a computed decomposition (tier A) and its
// basis / witness data (tier B).
void CollectDecomposition(Collector& C, const std::string& pt, const std::string& name,
                          const hecke::HeckeModuleDecomposition& D) {
    uint64_t h = FnvU64(kFnvOffset, D.blocks().size());
    uint64_t hw = FnvU64(kFnvOffset, D.blocks().size());
    for (const hecke::PrimaryBlock& b : D.blocks()) {
        h = FnvU64(h, b.dim);
        h = FnvU64(h, b.residue_degree);
        h = FnvU64(h, b.char_poly.size());
        for (const auto& kv : b.char_poly) { h = FnvU64(h, kv.first); h = HashPolyInto(h, kv.second); }
        for (const auto& kv : b.min_poly) { h = FnvU64(h, kv.first); h = HashPolyInto(h, kv.second); }
        for (const auto& kv : b.jordan_partition) {
            h = FnvU64(h, kv.first);
            h = FnvU64(h, kv.second.size());
            for (uint32_t part : kv.second) h = FnvU64(h, part);
        }
        hw = FnvU64(hw, b.residue_degree_certified ? 1 : 0);
        hw = FnvU64(hw, b.residue_witness.size());
        for (const auto& w : b.residue_witness) { hw = FnvU64(hw, w.first); hw = FnvU64(hw, w.second); }
    }
    C.raw(pt, name + "_invariants", 'A', D.blocks().size(), 0, h);
    C.scalar(pt, name + "_certify", 'A', D.certify());
    C.matrix(pt, name + "_U", 'B', D.change_of_basis());
    C.raw(pt, name + "_witness", 'B', D.blocks().size(), 0, hw);
    // (E0a) Both are no-ops on every ordinary golden run.
    if (g_dump) DumpDecomposition(pt, name, D);
    if (g_sweep) SweepDecomposition(C.records(), pt, name, D);
}

struct Timer {
    std::chrono::steady_clock::time_point t0 = std::chrono::steady_clock::now();
    double secs() const {
        return std::chrono::duration_cast<std::chrono::duration<double>>(
                   std::chrono::steady_clock::now() - t0).count();
    }
};

std::string PointNE(uint32_t n, uint32_t ell) {
    return "n=" + std::to_string(n) + " ell=" + std::to_string(ell);
}

// ===========================================================================
// (hecke-extensions slot 5, WP-3) THE EXTENSION-LAYER OBJECTS -- [X] Sec. 1.6.
//
// This is the stage's ONE add-only golden emission.  Everything below adds
// NEW (point, object) pairs; not one pre-existing entry is read, rewritten or
// re-ordered, which is why `golden_explain.py` must report 0 changed and 0
// removed lines.
//
// THE SUBGRID.  [X] Sec. 1.6: `n > 0` even and `5 <= ell < 2^32` -- outside it
// Eichler-Shimura and `d_2 = 0` do not hold ([XA M3]) and the ext layer is
// `uint32_t`-only.  On the shipped grid that is
//   n in {2, 4, 10, 16, 22, 24} x ell in {5, 13, 691, 10007, 134217757}
// plus the extra Dickson cell (32, 5) -- 31 cells.  EVERY other 32-bit cell
// and EVERY 64-bit point gets one tier-A text object `ext_excluded` naming the
// violated condition, exactly as the harness already records a shipped
// predicate's refusal (`excluded`, `dickson`).
//
// WHOSE COLLECTORS.  [X] Sec. 1.6's E3 / E4 / E5 names were shipped in slots 3
// and 4 DEFINED AND NEVER CALLED FROM HERE, by the unit that owns the object
// ([WP5-1] Sec. 10 / finding 5).  This file therefore CALLS them and does not
// re-spell them:
//   E3   HeckeExt/HeckeSignSplit::golden_objects()
//   E4b  HeckeExt/HeckeBoundary::golden_objects(p, nu)
//   E5-2 EquivariantMappingCone::golden_objects()
//   E5-3 Hecke/HeckeConeAction::golden_objects()
// The four families use OVERLAPPING object names (`rank_r^q` is both the
// boundary's and the cone's; `iota^q` is both the involution's and the cone's
// LES map), so each family gets its OWN POINT SUFFIX -- the reference is keyed
// by (point, object) and the harness already suffixes points (` p=<p> r=1`).
//
// The record's own objects (`ext_ideals`, `ext_certify`, `ext_json_canonical`,
// `ext_json_full`) are defined HERE, like every other golden recipe in this
// file (`CollectDecomposition`, `CollectCohomology`, ...): the golden harness
// is the single source of truth for a golden recipe.  The identically-shaped
// file-local functions in HeckeExt/HeckeExtRecordValidation.cpp are that
// validation's own stability oracle and are documented as such; they remain
// unreachable from here.
//
// `commit` IS REDACTED FROM BOTH JSON OBJECTS.  `hecke_ext/v2` carries a
// `commit` key baked from `git rev-parse HEAD` at CONFIGURE time
// (CMakeLists.txt, `OC_GIT_COMMIT`), whose own comment says no golden object
// reads it.  Hashing it would move two golden lines on every commit, so the
// canonical AND the full text are taken through `RedactJsonKeys` with
// `commit` in the list.  `ext_json_canonical` additionally redacts `t_p`, the
// stored restricted operators -- the only basis-dependent payload the record
// carries -- exactly as `report_canonical` redacts `level_b`.
//
// `settings` IS REDUCED TO ITS FINGERPRINT IN BOTH JSON OBJECTS (slot 7,
// WP-2, [X] Sec. 8 stage 1 (1)): ReduceSettingsToFingerprint, path-scoped to
// the top-level member, runs before the key redaction.  So registering a key
// at its default moves no golden line; a setting in force still does, through
// `settings.fingerprint`.  The one re-baseline this cost is the staged one's
// stage 1 (1) (62 changed `ext_json_*` lines, nothing added or removed).
//
// SETTINGS.  `u_ell = off` for the whole ext block ([X] Sec. 1.6: "the golden
// harness never builds U_ell"), installed by a scoped guard so the surrounding
// grid is untouched; and the family is `{T_2, T_3, T_5}`, the one the harness
// already decomposes at `decomp_sim{2,3,5}^1`.
// ===========================================================================

// n > 0, n even, 5 <= ell.  (`ell < 2^32` holds for the uint32_t grid.)
bool InExtSubgrid(uint32_t n, uint32_t ell) { return n > 0 && (n % 2 == 0) && ell >= 5; }

std::string ExtExclusionReason(uint32_t n, uint32_t ell) {
    if (n == 0) return "n_sym = 0 is outside the ext subgrid ([X] Sec. 1.6: n > 0)";
    if (n % 2 != 0) return "odd n_sym is outside the ext subgrid ([X] Sec. 1.6: n even)";
    if (ell < 5) return "ell < 5 is outside the ext subgrid ([X] Sec. 1.6: 5 <= ell < 2^32)";
    return "outside the ext subgrid";
}

// The [X] Sec. 1.6 `ext_ideals` recipe: ideal count, then per ideal the
// basis-independent identity fields and the verdict.
uint64_t HashExtIdeals(const hecke_ext::HeckeExtRecord& r) {
    uint64_t h = FnvU64(kFnvOffset, r.ideals.size());
    for (const hecke_ext::IdealRecord& id : r.ideals) {
        h = FnvU64(h, id.id);
        h = FnvU64(h, id.dim);
        h = FnvBytes(h, id.sign);
        h = FnvU64(h, id.residue_degree);
        h = FnvU64(h, id.residue_degree_certified ? 1u : 0u);
        h = FnvU64(h, id.local ? 1u : 0u);
        h = FnvU64(h, id.algebra_present ? 1u : 0u);
        h = FnvU64(h, id.algebra_dim);
        h = FnvU64(h, id.embdim);
        h = FnvU64(h, id.nilpotency);
        h = FnvU64(h, id.gorenstein ? 1u : 0u);
        h = FnvU64(h, id.hilbert.size());
        for (uint32_t v : id.hilbert) h = FnvU64(h, v);
        h = FnvBytes(h, id.certificate);
        h = FnvU64(h, static_cast<uint64_t>(id.final.value));
        h = FnvBytes(h, id.final.reason);
    }
    return h;
}

// The aggregate verify() code plus the per-ideal certificate routes and the
// Stage-5 structure verdicts -- [X] Sec. 1.6's `ext_certify`.
uint64_t HashExtCertify(const hecke_ext::HeckeExtRecord& r) {
    uint64_t h = FnvU64(kFnvOffset, r.verify());
    for (const hecke_ext::IdealRecord& id : r.ideals) {
        h = FnvBytes(h, id.certificate);
        h = FnvBytes(h, id.certificate_routes.a);
        h = FnvBytes(h, id.certificate_routes.b);
        h = FnvBytes(h, id.certificate_routes.c);
        for (const hecke_ext::ModuleRecordEntry& m : id.modules) h = FnvU64(h, m.structure.verify());
    }
    return h;
}

// [X] Sec. 1.6's `sturm_bound`: k, ell, dim M_k, the cap, B' (or "none"),
// closure_dim, and the Miller-basis matrices T_q mod ell for q <= B'.
uint64_t HashSturmBound(hecke_ext::SturmBound& B) {
    uint64_t h = FnvU64(kFnvOffset, B.weight());
    h = FnvU64(h, B.ell());
    h = FnvU64(h, B.dim_M_k());
    h = FnvU64(h, B.cap());
    h = FnvU64(h, B.has_B_prime() ? 1u : 0u);
    h = FnvU64(h, B.has_B_prime() ? B.B_prime() : 0u);
    h = FnvU64(h, B.closure_dim());
    h = FnvU64(h, B.primes_done().size());
    for (uint32_t q : B.primes_done()) {
        h = FnvU64(h, q);
        h = HashMatrixInto(h, B.T(q));
    }
    return h;
}

// One E3 / E4 / E5 collector family -> golden records.  `Obj` is any of the
// shipped POD collector structs (they agree field for field).  Every record
// is published under the tier ITS UNIT assigns: there is no harness-side tier
// override.  (Until hecke-extensions slot 6, WP-5, an `ExtTierFor` override
// mapped `A_c^...` -- the raw cone-operator matrix, which [X] Sec. 1.6 /
// owner decision OP7 put in tier B -- from the unit's 'A' to 'B'.  Its comment
// justified it by a digest "pinned at 0xedbff302058416f3"; that value is
// HeckeConeValidation's grid digest over EquivariantMappingCone::golden_digest()
// and contains no A_c, and HeckeConeAction::golden_digest() is printed
// nowhere -- so the justification was false.  WP-5 relabelled A_c^q 'B' in
// Hecke/HeckeConeAction::golden_objects() and removed the override; the
// published tier, hence GoldenReference.inc, is unchanged.)
template<class Obj>
void PushGolden(Collector& C, const std::string& pt, const std::vector<Obj>& objs) {
    for (const Obj& g : objs) {
        C.raw(pt, g.object, g.tier, g.rows, g.cols, g.hash);
    }
}

// The ext objects of one cell.  Never throws: a refusal by a shipped predicate
// is recorded as a text object, exactly like the surrounding grid's `excluded`.
void RunExtCell32(Collector& C, const SimplicialComplex& K,
                  const hecke::SL2ZTessellationTables& model, const std::string& pt,
                  uint32_t n, uint32_t ell) {
    if (!InExtSubgrid(n, ell)) {
        C.text(pt, "ext_excluded", 'A', ExtExclusionReason(n, ell));
        return;
    }
    Timer T;

    // [X] Sec. 1.6: the harness never builds U_ell.  Stamped `spec` (slot 7,
    // WP-2; owner confirmation 13): the harness is a spec, exactly as the
    // campaign that installs the same value from `u_ell = off` -- so the two
    // records agree on `source` too, not only after the stage-1 (1) reduction.
    hecke::UEllSetting u_off;
    u_off.mode = hecke::UEllMode::Off;
    const hecke::ScopedUEll u_guard(u_off, hecke::SettingSource::Spec);

    try {
        // --- Stages 1-5 + the two hecke_ext/v2 attachments ------------------
        hecke_ext::ExtDriverOptions opts;
        opts.good_primes    = {2, 3, 5};   // the family the harness decomposes
        opts.with_cone      = true;        // E4b / E5-2 record objects
        opts.with_sign_split = true;       // E3 record objects
        hecke_ext::HeckeExtDriver d(n, ell, opts);
        d.run();
        const hecke_ext::HeckeExtRecord& rec = d.record();

        const std::string pe = pt + " ext";
        C.raw(pe, "ext_ideals", 'A', rec.ideals.size(), 0, HashExtIdeals(rec));
        C.raw(pe, "ext_certify", 'A', rec.verify(), 0, HashExtCertify(rec));
        {
            const std::string j = NormalizeJsonWhitespace(hecke_ext::BuildHeckeExtReportJson(rec));
            // Slot-7 stage 1 (1): the top-level `settings` reduced to its
            // fingerprint (ReduceSettingsToFingerprint), THEN the key redaction.
            const std::string r = ReduceSettingsToFingerprint(j);
            const std::string full = RedactJsonKeys(r, {"commit"});
            const std::string can  = RedactJsonKeys(r, {"commit", "t_p"});
            C.text(pe, "ext_json_canonical", 'A', can);
            C.text(pe, "ext_json_full", 'B', full);
        }

        // --- the computed Sturm bound B'(k, ell), k = n + 2 -----------------
        {
            hecke_ext::SturmBound B(n + 2, ell);
            B.run();
            C.raw(pe, "sturm_bound", 'A', B.has_B_prime() ? B.B_prime() : 0u,
                  B.dim_M_k(), HashSturmBound(B));
        }

        // --- E3: the production iota^* and its per-degree sign split --------
        for (uint32_t q : d.degrees_run()) {
            const hecke_ext::HeckeSignSplit* s = d.sign_split(q);
            if (s == nullptr) continue;
            PushGolden(C, pt + " ext iota q=" + std::to_string(q), s->golden_objects());
        }

        // --- E5-2: the equivariant mapping cone ------------------------------
        if (d.cone() != nullptr) PushGolden(C, pt + " ext cone", d.cone()->golden_objects());

        // --- E4b: H^1_! = ker r^1 and the boundary Hecke action --------------
        // One pipeline, (p, nu) = (2, 1): the Eisenstein-line search and the
        // chi multiset both need p != ell, and 2 is in the subgrid's range for
        // every ell >= 5.
        const EquivariantCohomologyOnK& coh_K = d.boundary_pair()->bulk_cohomology();
        {
            hecke_ext::HeckeBoundaryOptions bopts;
            bopts.operators = {{2, 1}};
            hecke_ext::HeckeBoundary bd(&coh_K, K, model, bopts);
            bd.compute();
            PushGolden(C, pt + " ext bd p=2 nu=1", bd.golden_objects(2, 1));
        }

        // --- E5-3: the cone Hecke action A_c^{(q)}(T_2) ----------------------
        {
            hecke::HeckeCosetTables cosets(2, /*nu=*/1, ell, n);
            hecke::HeckeTauTable tau(K, model, cosets, ell, n);
            hecke::HeckeBoundaryTauTable taub(K, model, cosets);
            hecke::HeckePairHomotopy eta(tau, taub);
            hecke::HeckeOperatorOnK bulk(&coh_K, &cosets, &tau);
            bulk.compute();
            hecke::HeckeBoundaryOperator bdop(&d.boundary_pair()->boundary_cohomology(), &cosets, &taub);
            bdop.compute();
            hecke::HeckeConeAction act(d.cone(), &bulk, &bdop, &eta);
            act.compute();
            const std::string pa = pt + " ext cone_action";
            PushGolden(C, pa, act.golden_objects());
            // The diagnostics [X] Sec. 1.6 names in tier A.
            const hecke::ConeActionCertificate& ac = act.certificate();
            C.scalar(pa, "eta_zero", 'A', ac.eta_zero ? 1u : 0u);
            C.scalar(pa, "cochain_commutes", 'A', ac.cochain_commutes ? 1u : 0u);
            C.text(pa, "solve_regimes", 'A',
                   ac.solve_regime_eta + "/" + ac.solve_regime_tau + "/" +
                           ac.solve_regime_tau_boundary);
        }
    } catch (const std::exception& e) {
        // A shipped predicate refused this cell.  Recorded, like `excluded`.
        C.text(pt, "ext_excluded", 'A', std::string("ext layer refused: ") + e.what());
        std::cout << "  [" << pt << " ext] EXCLUDED: " << e.what() << "\n";
        return;
    }
    std::cout << "  [" << pt << " ext] " << std::fixed << std::setprecision(2) << T.secs() << " s\n";
}

// ===========================================================================
// (hecke-extensions slot 7, WP-1) THE E7 OBJECTS -- [X] Sec. 6.2 E7, stage 0.
//
// A SEPARATE add-only emission ([OD] S6-3): every record below is published
// under the NEW point suffix ` e7`, which no pre-existing line uses, so not
// one existing (point, object) key is read, rewritten or re-ordered, and the
// whole block runs LAST in the cell (after RunExtCell32).
//
// THE CELL SET is pinned to the ext subgrid of [X] Sec. 1.6 (InExtSubgrid:
// n > 0 even, 5 <= ell < 2^32 -- the 31 cells), which lies inside [C, cl:pd]'s
// range (ell >= 5, n even).  Off it NOTHING is emitted: the cell already
// carries `ext_excluded` naming the same violated condition.  A subgrid cell
// on which the E7 layer throws gets one tier-A `e7_excluded` text naming the
// reason, as `ext_excluded` does; no cell is excluded for cost.
//
// THE HECKE PRIMES are pinned to {2, 3, 5} = kGridP, the harness family
// ([S7A] m-8).
//
// WHOSE COLLECTORS.  Both families are DEFINED by their units (slot 6) and
// merely CALLED here:
//   Hecke/HeckeContragredient::ContragredientGoldenObjects
//       tier A dim_H_Dn^q, charpoly_A_Dn(T_p)^q        (q = 0..2, p in {2,3,5})
//   Hecke/HeckePairing::PairingGoldenObjects
//       tier A pairing_perfect^t, pd_adjoint^t; tier B pairing_matrix^t (t = 0..2)
// `pd_adjoint^t` is the AND over p in {2, 3, 5} of [C, cl:pd](2)'s
// adjointness A_c^{(t)T} G^t == G^t A(D_n)^{(2-t)} (factor 1: no determinant
// twist), computed exactly as HeckeDualityValidation's arm A12 computes it;
// t = 0 is vacuous (H^0_c = 0) and recorded as 1, as there.  These are
// RECORDED values: [C, cl:pd] is OPEN, and the harness freezes what the
// shipped code computes -- HeckeDualityValidation is the oracle that asserts.
// ===========================================================================
void RunE7Cell32(Collector& C, const std::string& pt, uint32_t n, uint32_t ell) {
    if (!InExtSubgrid(n, ell)) return;
    Timer T;
    const std::string pe = pt + " e7";
    try {
        const hecke::ContragredientCell cell(n, ell);
        const hecke::PoincareDualityPairing pd(cell);

        struct E7Op {
            hecke::ContragredientCell::HeckePair hp;
            std::unique_ptr<hecke::HeckeBoundaryTauTable> taub;
            std::unique_ptr<hecke::HeckePairHomotopy> eta;
            std::unique_ptr<hecke::HeckeBoundaryOperator> bd;
            std::unique_ptr<hecke::HeckeConeAction> act;
        };
        std::vector<std::unique_ptr<E7Op>> ops;
        std::vector<hecke::DualityHeckeInput> heckes;
        std::vector<char> adj_t(3, 1);
        for (uint64_t p64 : kGridP) {
            const uint32_t p = static_cast<uint32_t>(p64);
            auto o = std::make_unique<E7Op>();
            o->hp = cell.hecke_pair(p);
            o->taub = std::make_unique<hecke::HeckeBoundaryTauTable>(cell.K(), cell.model(), *o->hp.cosets);
            o->eta = std::make_unique<hecke::HeckePairHomotopy>(*o->hp.tau, *o->taub);
            o->bd = std::make_unique<hecke::HeckeBoundaryOperator>(&pd.pair_M().boundary_cohomology(),
                                                                   o->hp.cosets.get(), o->taub.get());
            o->bd->compute();
            o->act = std::make_unique<hecke::HeckeConeAction>(&pd.cone(), o->hp.op_M.get(), o->bd.get(),
                                                              o->eta.get());
            o->act->compute();
            const auto& A_D = o->hp.op_D->get_H_matrices();
            ENGINE_CHECK_MSG(A_D.size() == 3, "RunE7Cell32: A(D_n)(T_p) must have three degrees");
            for (uint32_t t = 1; t < 3; ++t)
                if (!hecke::PairingAdjoint(o->act->A_c(t), pd.gram(t), A_D[2 - t], 1u, ell)) adj_t[t] = 0;
            heckes.push_back(hecke::DualityHeckeInput{ p, &o->hp.op_D->get_H_matrices() });
            ops.push_back(std::move(o));
        }
        PushGolden(C, pe, hecke::ContragredientGoldenObjects(cell.coh_D(), heckes));
        PushGolden(C, pe, hecke::PairingGoldenObjects(pd, adj_t));
    } catch (const std::exception& e) {
        C.text(pt, "e7_excluded", 'A', std::string("E7 layer refused: ") + e.what());
        std::cout << "  [" << pe << "] EXCLUDED: " << e.what() << "\n";
        return;
    }
    std::cout << "  [" << pe << "] " << std::fixed << std::setprecision(2) << T.secs() << " s\n";
}

// One (n, ell) cell of the 32-bit grid: cohomology objects + all Hecke objects.
void RunCell32(Collector& C, const SimplicialComplex& K, const hecke::SL2ZTessellationTables& model,
               uint32_t n, uint32_t ell) {
    const std::string pt = PointNE(n, ell);
    Timer T;
    C.scalar(pt, "dickson", 'A', IsDicksonExceptional(n, ell) ? 1 : 0);

    std::unique_ptr<LocalSystemSimplicialComplex> lssc;
    std::unique_ptr<EquivariantCohomologyOnK> coh;
    try {
        lssc = std::make_unique<LocalSystemSimplicialComplex>(K, BuildRelations(K, n, ell), n + 1, ell);
        coh = std::make_unique<EquivariantCohomologyOnK>(lssc.get());
        coh->compute_cohomology();
    } catch (const std::invalid_argument& e) {
        // The shipped predicates reject this point: record and skip.
        C.text(pt, "excluded", 'A', e.what());
        std::cout << "  [" << pt << "] EXCLUDED by shipped predicate: " << e.what() << "\n";
        // (slot 5, WP-3) The ext layer sits on this cohomology, so a cell the
        // shipped predicate rejects is ext-excluded too.  Unreachable on the
        // shipped grid (no cell is rejected), kept so the rule is total.
        C.text(pt, "ext_excluded", 'A', std::string("bulk pipeline excluded: ") + e.what());
        return;
    }
    CollectCohomology(C, pt, *coh);
    const double t_coh = T.secs();

    hecke::HeckeOperatorFamily fam(coh.get(), K, model, ell, n);
    for (uint64_t p : kGridP) {
        const std::string ptp = pt + " p=" + std::to_string(p);
        try {
            // --- r = 1: one explicit pipeline (Theta_flat, Theta_eq, A, op JSON). ---
            hecke::HeckeCosetTables cosets(p, /*r=*/1, ell, n);
            hecke::HeckeTauTable tau(K, model, cosets, ell, n);
            hecke::HeckeOperatorOnK op(coh.get(), &cosets, &tau);
            op.compute();
            const auto& Tf = op.get_Theta_flat_matrices();
            const auto& Te = op.get_Theta_eq_matrices();
            for (uint32_t q = 0; q < Tf.size(); ++q) {
                C.flat(ptp + " r=1", "Theta_flat" + Q(q), 'A', Tf[q]);
                FlatMatrix rk = Tf[q];
                C.scalar(ptp + " r=1", "rank_Theta_flat" + Q(q), 'A',
                         (rk.get_rows() == 0 || rk.get_cols() == 0) ? 0 : rk.rank());
                if (q < Te.size()) C.matrix(ptp + " r=1", "Theta_eq" + Q(q), 'B', Te[q]);
            }
            CollectOperatorMatrices(C, ptp + " r=1", "A(T_p)", op.get_H_matrices());
            C.text(ptp + " r=1", "op_json_full", 'B', NormalizeJsonWhitespace(op.to_json()));

            // --- the family: recursion / direct / primitive / diamond. ---
            const std::vector<Mat32>& Ap = fam.T_p(p);
            // Cross-check the family's r=1 run reproduces the explicit one (it must:
            // same code path). Recorded, not asserted here -- the hash pins it.
            uint64_t hcmp = kFnvOffset;
            for (uint32_t q = 0; q < Ap.size(); ++q) hcmp = HashMatrixInto(hcmp, Ap[q]);
            C.raw(ptp + " r=1", "A(T_p)_via_family", 'B', Ap.size(), 0, hcmp);

            CollectOperatorMatrices(C, ptp + " r=2", "A(T_p^2)_rec", fam.T_p_pow(p, 2));
            CollectOperatorMatrices(C, ptp + " r=2", "A(T_p^2)_dir", fam.T_p_pow_direct(p, 2));
            CollectOperatorMatrices(C, ptp + " r=2", "A'(T_p^2)_prim", fam.T_p_pow_primitive_direct(p, 2));
            {
                // The diamond scalar p^n I is canonical as a matrix: raw hash is tier A.
                const std::vector<Mat32> Tpp = fam.T_pp(p);
                for (uint32_t q = 0; q < Tpp.size(); ++q) C.matrix(ptp + " r=1", "T(p,p)" + Q(q), 'A', Tpp[q]);
            }

            // --- level (c) + report, per degree with d_q > 0. ---
            const auto& A = op.get_H_matrices();
            for (uint32_t q = 0; q < A.size(); ++q) {
                if (A[q].rows == 0) continue;
                std::vector<std::pair<uint64_t, Mat32>> ops = {{p, A[q]}};
                hecke::HeckeModuleDecomposition D(ops, ell);
                D.compute();
                CollectDecomposition(C, ptp + " r=1", "decomp_A(T_p)" + Q(q), D);

                hecke::HeckeReportInput in;
                in.ell = ell;
                in.n_sym = n;
                in.degree = q;
                // (E0b, slot 2 WP-2) The four oc1a record fields are DERIVED
                // from the complex's own degree count -- the very `A.size()`
                // this loop runs over, which is `num_degrees` at :469 -- so
                // that E9's truncated model reports its own top degree and no
                // literal `2` is ever written.
                in.num_degrees = static_cast<uint32_t>(A.size());
                in.level_b = {{p, A[q]}};
                in.decomp = &D;
                in.level_a_theta_flat = {{p, Tf[q]}};
                const std::string json = NormalizeJsonWhitespace(hecke::BuildHeckeReportJson(in));
                C.text(ptp + " r=1", "report_full" + Q(q), 'B', json);
                C.text(ptp + " r=1", "report_canonical" + Q(q), 'A',
                       RedactJsonKeys(json, {"level_b", "basis_columns", "change_of_basis"}));
            }
        } catch (const std::invalid_argument& e) {
            C.text(ptp, "excluded", 'A', e.what());
            std::cout << "  [" << ptp << "] EXCLUDED by shipped predicate: " << e.what() << "\n";
        }
    }

    // Simultaneous decomposition of {T_2, T_3, T_5} at q = 1 (when H^1 != 0).
    {
        const auto& dims = coh->get_H_eq_dimensions();
        if (dims.size() > 1 && dims[1] > 0) {
            std::vector<std::pair<uint64_t, Mat32>> ops;
            for (uint64_t p : kGridP) ops.emplace_back(p, fam.T_p(p)[1]);
            hecke::HeckeModuleDecomposition D(ops, ell);
            D.compute();
            CollectDecomposition(C, pt, "decomp_sim{2,3,5}^1", D);
        }
    }

    std::cout << "  [" << pt << "] dims=(";
    const auto& d = coh->get_H_eq_dimensions();
    for (size_t i = 0; i < d.size(); ++i) std::cout << (i ? "," : "") << d[i];
    std::cout << ") cohomology " << std::fixed << std::setprecision(2) << t_coh << " s, total "
              << T.secs() << " s\n";

    // (slot 5, WP-3) The extension-layer objects of [X] Sec. 1.6.  LAST in the
    // cell, so every pre-existing record of this cell keeps its emission
    // position and the new lines are pure insertions.
    RunExtCell32(C, K, model, pt, n, ell);

    // (slot 7, WP-1) The E7 objects, stage 0: LAST in the cell, after the ext
    // block, for the same reason -- pure insertions.
    RunE7Cell32(C, pt, n, ell);
}

// One 64-bit point (HeckeExtendedRegimeValidation recipe).
void RunCell64(Collector& C, const SimplicialComplex& K, const hecke::SL2ZTessellationTables& model,
               uint64_t ell, uint32_t n) {
    const std::string pt = "n=" + std::to_string(n) + " ell=" + std::to_string(ell) + " [u64]";
    Timer T;
    auto rels = GenerateSL2ZRelationsElt<uint64_t>(K, n, ell);
    LocalSystemSimplicialComplex64 lssc(K, rels, n + 1, ell);
    EquivariantCohomologyOnK64 coh(&lssc);
    coh.compute_cohomology();
    CollectCohomology(C, pt, coh);
    for (uint64_t p : kGridP64) {
        const std::string ptp = pt + " p=" + std::to_string(p) + " r=1";
        hecke::HeckeCosetTables64 cosets(p, /*r=*/1, ell, n);
        hecke::HeckeTauTable64 tau(K, model, cosets, ell, n);
        hecke::HeckeOperatorOnK64 op(&coh, &cosets, &tau);
        op.compute();
        const auto& Tf = op.get_Theta_flat_matrices();
        const auto& Te = op.get_Theta_eq_matrices();
        for (uint32_t q = 0; q < Tf.size(); ++q) {
            C.flat(ptp, "Theta_flat" + Q(q), 'A', Tf[q]);
            FlatMatrix64 rk = Tf[q];
            C.scalar(ptp, "rank_Theta_flat" + Q(q), 'A',
                     (rk.get_rows() == 0 || rk.get_cols() == 0) ? 0 : rk.rank());
            if (q < Te.size()) C.matrix(ptp, "Theta_eq" + Q(q), 'B', Te[q]);
        }
        CollectOperatorMatrices(C, ptp, "A(T_p)", op.get_H_matrices());
    }
    // (slot 5, WP-3) [X] Sec. 1.6: the ext subgrid is `5 <= ell < 2^32` and the
    // extension layer is uint32_t-only, so EVERY 64-bit point is ext-excluded,
    // recorded as one tier-A text object.
    C.text(pt, "ext_excluded", 'A',
           "64-bit point: the extension layer is uint32_t-only ([X] Sec. 1.6: ell < 2^32)");
    std::cout << "  [" << pt << "] " << std::fixed << std::setprecision(2) << T.secs() << " s\n";
}

// S6 induced-map cases: the public-API call sequences of
// HeckeMorphismRegressionValidation part 1 (replicated; those builders live in an
// anonymous namespace).
void CollectInduced(Collector& C, const std::string& pt, const std::vector<FlatMatrix>& maps) {
    C.scalar(pt, "num_degrees", 'A', maps.size());
    for (uint32_t q = 0; q < maps.size(); ++q) {
        // (matrix-engine Phase 3) The induced maps are FlatMatrix now; Rank and
        // CharPoly still take a MatrixT, and these are 1x1..3x3, so the copy is
        // free. C.matrix's FlatMatrixT overload hashes the same bytes.
        const Mat32 m = maps[q].to_matrix();
        C.scalar(pt, "rank_induced" + Q(q), 'A', Rank(m));
        if (maps[q].get_rows() == maps[q].get_cols()) C.poly(pt, "charpoly_induced" + Q(q), 'A', CharPoly(m));
        C.matrix(pt, "induced" + Q(q), 'B', maps[q]);
    }
}

void RunS6Presets(Collector& C) {
    const uint32_t p = 5;
    LocalSystemSimplicialComplex lssc_triv = LocalSystemPresets::CreateS1Scalar(1, p, "_morph_test_triv");
    EquivariantCohomologyOnK eq_triv(&lssc_triv);
    eq_triv.compute_cohomology();
    LocalSystemSimplicialComplex lssc_unip = LocalSystemPresets::CreateS1Unipotent2D(p, "_morph_test_unip");
    EquivariantCohomologyOnK eq_unip(&lssc_unip);
    eq_unip.compute_cohomology();
    {
        EquivariantCochainComplexMorphismOnK m =
            LocalSystemPresets::CreateS1_UnipotentToTrivialMorphism(&eq_triv, &eq_unip);
        m.compute_induced_maps_on_cohomology();
        CollectInduced(C, "S6 preset_unip_to_triv p=5", m.get_induced_H_eq_maps());
    }
    {
        EquivariantCochainComplexMorphismOnK m =
            LocalSystemPresets::CreateS1_TrivialToUnipotentMorphism(&eq_unip, &eq_triv);
        m.compute_induced_maps_on_cohomology();
        CollectInduced(C, "S6 preset_triv_to_unip p=5", m.get_induced_H_eq_maps());
    }
    {
        EquivariantCochainComplexMorphismOnK m = LocalSystemPresets::CreateS1_IdentityMorphism(&eq_unip);
        m.compute_induced_maps_on_cohomology();
        CollectInduced(C, "S6 preset_identity_unip p=5", m.get_induced_H_eq_maps());
    }
}

void RunS6Restriction(Collector& C, const SimplicialComplex& K_sl2z, uint32_t n_sym, uint32_t ell) {
    const std::string pt = "S6 sl2z_restrict n=" + std::to_string(n_sym) + " ell=" + std::to_string(ell);
    Timer T;
    const uint32_t d_V = n_sym + 1;
    const FlatMatrix T0(2, 2, ell, std::vector<uint32_t>{1, 1, 0, 1});
    const FlatMatrix M_T = ComputeSymmetricPowerMatrixFM(T0, n_sym);

    LocalSystemSimplicialComplex lssc_sl2z(K_sl2z, BuildRelations(K_sl2z, n_sym, ell), d_V, ell);
    EquivariantCohomologyOnK eq_sl2z(&lssc_sl2z);
    eq_sl2z.compute_cohomology();

    std::vector<std::vector<std::vector<uint32_t>>> path = {
        {{0}, {1}, {2}, {3}},
        {make_edge(0, 1), make_edge(1, 2), make_edge(2, 3)}};
    SimplicialComplex K_path(path, "K_S1_Path_Repr_For_SL2Z");
    LocalSystemSimplicialComplex lssc_cusp(
        K_path,
        {LocalSystemSimplicialComplex::SimplexRelation(0, find_simplex_idx(K_path, 0, {3}),
                                                       find_simplex_idx(K_path, 0, {0}), M_T,
                                                       "S1_cusp_loop_on_path_for_morph")},
        d_V, ell);
    EquivariantCohomologyOnK eq_cusp(&lssc_cusp);
    eq_cusp.compute_cohomology();

    SimplicialComplexMorphism scm(&K_path, &K_sl2z);
    if (!scm.is_valid()) throw std::runtime_error(pt + ": SimplicialComplexMorphism invalid");
    const FlatMatrix F_identity(d_V, ell);
    EquivariantCochainComplexMorphismOnK restr(&eq_cusp, &eq_sl2z, &scm, F_identity);
    restr.compute_induced_maps_on_cohomology();
    CollectInduced(C, pt, restr.get_induced_H_eq_maps());
    std::cout << "  [" << pt << "] " << std::fixed << std::setprecision(2) << T.secs() << " s\n";
}

// ===========================================================================
// (hecke-extensions slot 8, WP-4; E8 item 4a) THE E8 OBJECTS -- [E] v1.9.0-pre.1
// sec:ring (d)-(e).  One NEW point per kGridE8 entry, emitting at that point
// only (no marker anywhere else; an object undefined at an uncertified level
// is simply absent):
//   tier A  lengths_H^q, invariant_factors_H^q (q = 0, 1, 2),
//           r_K(d^0), r_K(d^1), ladic_type_certified,
//           [certified only] ladic_type^q (q = 0, 1, 2),
//           bockstein_rank^0, higher_torsion_count,
//           [certified only] charpoly_A(T_p)^1_free_p<p> for each of the point's primes;
//   tier B  red_H^q (q = 0, 1, 2): the reduction H^q(Z/ell^k) -> H^q(F_ell), the
//           (c5) matrix from the cyclic-summand basis to the D5b basis.
// Recipes: a scalar stores its value (the harness's scalar convention); a list
// (invariant_factors, ladic_type) hashes its length then its entries (ladic_type:
// free rank first), rows = its length (+1 for ladic_type), cols = 1; the free
// charpoly hashes its precision K - e_max, then its coefficients low -> high,
// rows = the number of coefficients, cols = the precision (the precision is part
// of the object); red_H^q hashes rows, cols, then the entries row-major, exactly
// as a matrix.  bockstein_rank^0 / higher_torsion_count are mod-ell objects, the
// same at every k of a cell: they are computed from the cell's Z/ell^2 object.
//
// The ring (`coefficient_ring = Z/ell^<k>`, source spec) is installed by
// ScopedMathSettings inside BuildE8RingCell -- i.e. inside RunE8Point, around
// the construction of the E8 objects only (the `u_off` precedent of
// RunExtCell32); the F_ell comparison object is built outside it, at the
// defaults, and every other arm of the harness runs at the defaults.
// ===========================================================================
struct E8RingCell {
    std::unique_ptr<LocalSystemSimplicialComplex> lssc;
    std::unique_ptr<EquivariantCohomologyOnK> coh;
    // (slot-9 WP-4, E8 item 4b) the pair and the cone over the same ring.
    std::unique_ptr<hecke::SL2ZBoundaryPair> pair;
    std::unique_ptr<EquivariantMappingCone> cone;
    std::vector<std::unique_ptr<hecke::HeckeCosetTables>> cosets;
    std::vector<std::unique_ptr<hecke::HeckeTauTable>> tau;
    std::vector<std::unique_ptr<hecke::HeckeOperatorOnK>> ops;
};

// Z/ell^k installed for the lifetime of one scope (the object reads it once, at
// construction, and keeps it).
hecke::MathSettings E8RingSettings(uint32_t k) {
    hecke::MathSettings s = hecke::math_settings();
    s.coefficient_ring.k = k;
    return s;
}

std::unique_ptr<E8RingCell> BuildE8RingCell(const SimplicialComplex& K, uint32_t n, uint32_t ell, uint32_t k) {
    const hecke::ScopedMathSettings ring(E8RingSettings(k), hecke::SettingSource::Spec);
    auto c = std::make_unique<E8RingCell>();
    const engine::ModCtx<uint32_t> ctx = hecke::CoefficientRingContext<uint32_t>("GoldenHarness E8", ell, k);
    c->lssc = std::make_unique<LocalSystemSimplicialComplex>(K, hecke::GenerateSL2ZRelationsRing<uint32_t>(K, n, ctx),
                                                             n + 1, ctx.p);
    c->coh = std::make_unique<EquivariantCohomologyOnK>(c->lssc.get());
    c->coh->compute_cohomology();
    // (slot-9 WP-4, E8 item 4b) the boundary and the cone over Z/ell^k (their
    // dense module route; constructed under the same ring scope).
    c->pair = std::make_unique<hecke::SL2ZBoundaryPair>(c->coh.get(), K);
    c->cone = c->pair->make_cone();
    return c;
}

// A(T_p) over Z/ell^k on the cell's cohomology, one operator per prime.
void AddE8Operators(E8RingCell& c, const SimplicialComplex& K, const hecke::SL2ZTessellationTables& model,
                    uint32_t n, uint32_t ell, uint32_t k, const std::vector<uint64_t>& primes) {
    const hecke::ScopedMathSettings ring(E8RingSettings(k), hecke::SettingSource::Spec);
    for (uint64_t p : primes) {
        c.cosets.push_back(std::make_unique<hecke::HeckeCosetTables>(p, /*r=*/1, ell, n));
        c.tau.push_back(std::make_unique<hecke::HeckeTauTable>(K, model, *c.cosets.back(), ell, n));
        c.ops.push_back(std::make_unique<hecke::HeckeOperatorOnK>(c.coh.get(), c.cosets.back().get(),
                                                                    c.tau.back().get()));
        c.ops.back()->compute();
    }
}

uint64_t HashU32List(uint64_t h, const std::vector<uint32_t>& v) {
    h = FnvU64(h, v.size());
    for (uint32_t e : v) h = FnvU64(h, e);
    return h;
}

void RunE8Point(Collector& C, const SimplicialComplex& K, const hecke::SL2ZTessellationTables& model,
                const E8Point& e8) {
    const uint32_t n = e8.n, ell = e8.ell, k = e8.k;
    const std::string pt = "E8 ring=Z/" + std::to_string(ell) + "^" + std::to_string(k) + " n=" + std::to_string(n) +
                           " ell=" + std::to_string(ell);
    Timer T;
    // The F_ell comparison object, at the defaults (a field object is refused under Z/ell^k).
    LocalSystemSimplicialComplex lssc_f(K, BuildRelations(K, n, ell), n + 1, ell);
    EquivariantCohomologyOnK coh_f(&lssc_f);
    coh_f.compute_cohomology();

    // The E8 objects over Z/ell^k.  The Hecke operators only at a certified
    // level: the free charpoly is defined nowhere else (M-1).
    const std::unique_ptr<E8RingCell> rk = BuildE8RingCell(K, n, ell, k);
    const bool cert = rk->coh->ladic_type_certified();
    if (cert) AddE8Operators(*rk, K, model, n, ell, k, e8.primes);
    const EquivariantCohomologyOnK& c = *rk->coh;
    // The Bockstein is a mod-ell object: from the cell's Z/ell^2 object.
    std::unique_ptr<E8RingCell> r2;
    if (k != 2) r2 = BuildE8RingCell(K, n, ell, 2);
    const EquivariantCohomologyOnK& c2 = (k == 2) ? c : *r2->coh;

    const std::vector<uint32_t> len = c.H_lengths();
    for (uint32_t q = 0; q < 3; ++q) C.scalar(pt, "lengths_H" + Q(q), 'A', len[q]);
    for (uint32_t q = 0; q < 3; ++q) {
        const std::vector<uint32_t>& e = c.H_invariant_factors(q);
        C.raw(pt, "invariant_factors_H" + Q(q), 'A', e.size(), 1, HashU32List(kFnvOffset, e));
    }
    C.scalar(pt, "r_K(d^0)", 'A', c.r_K(0));
    C.scalar(pt, "r_K(d^1)", 'A', c.r_K(1));
    C.scalar(pt, "ladic_type_certified", 'A', cert ? 1 : 0);
    if (cert) {
        for (uint32_t q = 0; q < 3; ++q) {
            const auto t = c.ladic_type(q);
            C.raw(pt, "ladic_type" + Q(q), 'A', t.torsion.size() + 1, 1,
                  HashU32List(FnvU64(kFnvOffset, t.free_rank), t.torsion));
        }
    }
    const hecke::BocksteinBookkeeping bk = hecke::BocksteinCounts(c2, coh_f);
    C.scalar(pt, "bockstein_rank^0", 'A', bk.bockstein_rank0);
    C.scalar(pt, "higher_torsion_count", 'A', bk.higher_torsion_count);
    if (cert) {
        for (size_t i = 0; i < e8.primes.size(); ++i) {
            const auto fc = rk->ops[i]->free_charpoly_H1();
            uint64_t h = FnvU64(kFnvOffset, fc.precision);
            h = FnvU64(h, fc.coeffs.size());
            for (uint32_t a : fc.coeffs) h = FnvU64(h, a);
            C.raw(pt, "charpoly_A(T_p)^1_free_p" + std::to_string(e8.primes[i]), 'A', fc.coeffs.size(), fc.precision, h);
        }
    }
    for (uint32_t q = 0; q < 3; ++q) {
        const engine::DenseBuf<uint32_t> R = hecke::ReduceHToField(c, coh_f, q);
        uint64_t h = FnvU64(kFnvOffset, R.rows);
        h = FnvU64(h, R.cols);
        for (uint32_t i = 0; i < R.rows; ++i)
            for (uint32_t j = 0; j < R.cols; ++j) h = FnvU64(h, R.at(i, j));
        C.raw(pt, "red_H" + Q(q), 'B', R.rows, R.cols, h);
    }

    // (slot-9 WP-4; E8 item 4b; [E] v1.10.0-pre.1 sec:ring (f)) THE BOUNDARY AND
    // THE CONE, after the E8 item 4a objects, in (f)'s order.  Tier A: the
    // boundary's lengths and invariant factors (q = 0, 1), its mod-ell Bockstein
    // rank (from the cell's Z/ell^2 boundary), the cone's lengths and invariant
    // factors (q = 0, 1, 2, in the H_c indexing), coneexact_iv_certified (a
    // failure aborts, so it is 1 wherever emitted) and the eight image lengths
    // of eq:coneles; tier B: the reductions to F_ell of the boundary and the
    // cone (the F_ell pair and cone at the defaults).  Recipes as above: a
    // scalar its value, a list its length then its entries (rows = length,
    // cols = 1), a reduction rows, cols, entries row-major (0 x 0 legal).
    const hecke::SL2ZBoundaryPair pair_f(&coh_f, K);
    const std::unique_ptr<EquivariantMappingCone> cone_f = pair_f.make_cone();
    const EquivariantCohomologyOnK& bd = rk->pair->boundary_cohomology();
    const ModuleConeT<uint32_t>& mc = rk->cone->module_cone();
    const std::vector<uint32_t> blen = bd.H_lengths();
    for (uint32_t q = 0; q < 2; ++q) C.scalar(pt, "bdry_lengths_H" + Q(q), 'A', blen[q]);
    for (uint32_t q = 0; q < 2; ++q) {
        const std::vector<uint32_t>& e = bd.H_invariant_factors(q);
        C.raw(pt, "bdry_invariant_factors_H" + Q(q), 'A', e.size(), 1, HashU32List(kFnvOffset, e));
    }
    const EquivariantCohomologyOnK& bd2 = (k == 2) ? bd : r2->pair->boundary_cohomology();
    const uint32_t bdry_beta = hecke::RankOverField<uint32_t>(ell, hecke::BocksteinH(bd2, pair_f.boundary_cohomology(), 0));
    C.scalar(pt, "bdry_bockstein_rank^0", 'A', bdry_beta);
    const std::vector<uint32_t> clen = mc.H_lengths();
    for (uint32_t q = 0; q < 3; ++q) C.scalar(pt, "cone_lengths_Hc" + Q(q), 'A', clen[q]);
    for (uint32_t q = 0; q < 3; ++q) {
        const std::vector<uint32_t>& e = mc.H_invariant_factors(q);
        C.raw(pt, "cone_invariant_factors_Hc" + Q(q), 'A', e.size(), 1, HashU32List(kFnvOffset, e));
    }
    C.scalar(pt, "coneexact_iv_certified", 'A', mc.coneexact_iv_certified() ? 1 : 0);
    {
        const std::vector<uint32_t> im = mc.les_im_lengths();
        C.raw(pt, "cone_les_im_lengths", 'A', im.size(), 1, HashU32List(kFnvOffset, im));
    }
    auto raw_matrix = [&](const std::string& name, const engine::DenseBuf<uint32_t>& R) {
        uint64_t h = FnvU64(kFnvOffset, R.rows);
        h = FnvU64(h, R.cols);
        for (uint32_t i = 0; i < R.rows; ++i)
            for (uint32_t j = 0; j < R.cols; ++j) h = FnvU64(h, R.at(i, j));
        C.raw(pt, name, 'B', R.rows, R.cols, h);
    };
    for (uint32_t q = 0; q < 2; ++q)
        raw_matrix("bdry_red_H" + Q(q), hecke::ReduceHToField(bd, pair_f.boundary_cohomology(), q));
    for (uint32_t q = 0; q < 3; ++q) raw_matrix("cone_red_Hc" + Q(q), hecke::ReduceConeHcToField(mc, *cone_f, q));

    std::cout << "  [" << pt << "] certified=" << (cert ? 1 : 0) << " " << std::fixed << std::setprecision(2)
              << T.secs() << " s\n";
}

// ===========================================================================
// (hecke-extensions slot 13, WP-5; owner Q5 = yes) THE HIDA POINTS -- the first
// Hida golden emission, ADD-ONLY: the slot-12 ledger's candidate list
// (docs/implementation/slot12/golden_ledger.md), at NEW points
// `HIDA n=<n> ell=<ell>`, looped from ComputeAll() ONLY (after the E8 loop),
// never from RunCell32 -- so GoldenSubset32 (EngineDeterminismValidation), the ext
// records (HeckeExtDeterminismValidation) and the console C4 (a) scripts cannot
// reach them.  Every object is tier A (basis-independent):
//   prepass_e_max, prepass_torsion  -- [C] prop:torsionprepass: e_max and the
//        torsion multiset {a_t} (descending) of H^1(Gamma; Sym^n Z^2) (x) Z_ell,
//        from J_n alone (Hecke/HeckeOrdinaryPart); the list hashes its length
//        then its entries (HashU32List), rows = its length, cols = 1;
//   ord_rank_He<e'> -- [C] lem:ordexponent, cor:ordprecision(1): the F_ell-rank of
//        U_ell^{N_ord} on H_{e'} = H^1(Gamma; Sym^n (Z/ell^{e'})^2), read on
//        H_K / ell^{e'} at K = e'_max + e_max (HeckeExt/HeckeHidaTower::ReadOnModule);
//   chi_ord_T<p>_mod_ell2 -- [C] def:hidatower: the characteristic polynomial of
//        T_p on the free Z/ell^2-module e_ord H_2 (the same reading), hashed as
//        the E8 free charpoly: the precision (2) then the coefficients low ->
//        high; rows = the number of coefficients, cols = the precision.
// The ring scope is HeckeHidaTower's own (ScopedCoefficientRing inside
// RunModuleCell); every other arm of the harness runs at the defaults.
// ===========================================================================
struct HidaPoint {
    uint32_t n;
    uint64_t ell;
    bool prepass;
    std::vector<uint32_t> rank_levels;   // e' for ord_rank_He<e'>
    bool chi_mod_ell2;                   // chi_ord_T2 / _T3 mod ell^2
};
const std::vector<HidaPoint> kGridHida = {
    {6, 5, true, {}, false},
    {10, 11, false, {}, true},
    {10, 691, true, {2}, false},
    {16, 5, true, {}, false},
    {30, 5, true, {2, 3}, false},
    {56, 7, true, {}, false},
    {60, 5, true, {}, false},
    {120, 11, false, {}, true},
    {150, 5, true, {}, false},
};

void RunHidaPoint(Collector& C, const HidaPoint& hp) {
    namespace hida = hecke_ext::hida;
    const std::string pt = "HIDA n=" + std::to_string(hp.n) + " ell=" + std::to_string(hp.ell);
    Timer T;
    const hecke::ordinary::TorsionPrepass pre = hecke::ordinary::TorsionExponentPrepass(hp.n, hp.ell);
    if (hp.prepass) {
        C.scalar(pt, "prepass_e_max", 'A', pre.e_max);
        C.raw(pt, "prepass_torsion", 'A', pre.torsion.size(), 1, HashU32List(kFnvOffset, pre.torsion));
    }
    uint32_t m_max = hp.chi_mod_ell2 ? 2 : 0;
    for (uint32_t e : hp.rank_levels) m_max = std::max(m_max, e);
    if (m_max > 0) {
        hecke::ordinary::RequirePrecisionUse(hp.n, hp.ell, m_max);
        const uint32_t K = m_max + pre.e_max;   // cor:ordprecision(2)
        const std::vector<uint64_t> primes = hp.chi_mod_ell2 ? std::vector<uint64_t>{hp.ell, 2, 3}
                                                             : std::vector<uint64_t>{hp.ell};
        const hida::ModuleCell mc = hida::RunModuleCell(hp.n, hp.ell, std::max<uint32_t>(2, K), primes);
        for (uint32_t e : hp.rank_levels)
            C.scalar(pt, "ord_rank_He" + std::to_string(e), 'A', hida::ReadOnModule(mc, e).ord_rank);
        if (hp.chi_mod_ell2) {
            const hida::Reading r2 = hida::ReadOnModule(mc, 2);
            for (size_t i = 0; i < r2.primes.size(); ++i) {
                uint64_t h = FnvU64(kFnvOffset, 2);
                for (uint64_t c : r2.charpoly[i]) h = FnvU64(h, c);
                C.raw(pt, "chi_ord_T" + std::to_string(r2.primes[i]) + "_mod_ell2", 'A', r2.charpoly[i].size(), 2, h);
            }
        }
    }
    std::cout << "  [" << pt << "] " << std::fixed << std::setprecision(2) << T.secs() << " s\n";
}

// ===========================================================================
// (hecke-extensions slot 17, WP-4; owner O2 of 2026-10-01 = yes, the event S16-O3
// batched) THE LEVELN, LOEWY_RING AND NEB POINTS -- one ADD-ONLY emission at NEW
// points, looped from ComputeAll() ONLY (after the HIDA loop), never from RunCell32
// -- so GoldenSubset32 (EngineDeterminismValidation), the ext records
// (HeckeExtDeterminismValidation) and the console C4 (a) scripts cannot reach them.
// The expect set (docs/matrix-engine/rebaseline-2026-10-02-addonly-levelN-loewy-neb/
// expect.tsv) was committed before the emission.  The recipes:
//
// `LEVELN N=<N> n=<n> p=1000003` ([C] sec:levelN, sec:levelNdegsec; over F_p, p the
//   characteristic-0 surrogate field of every level-N validation):
//   h0, h1                   tier A scalars of the sparse presentation M_N / Rel_N
//                            (prop:levelorbit (2); SparseCokernel::Build);
//   charpoly_{T,U}_q^1       tier A, q in {2, 3, 5, 7} (U_q iff q | N): the charpoly of
//                            [H_{N,q}] on M_N / Rel_N (prop:heilbronn;
//                            CokernelHeckeMatrices, HCharpoly low -> high), hashed
//                            FNV(#coeffs, coeffs); rows = #coeffs, cols = 1;
//   H_{T,U}_q^1              tier B: that matrix in the natural-RREF free-column basis,
//                            FNV(rows, cols, entries row-major); dims rows x cols;
//   model_dim_H^1            tier A: dim H^1 of the fixed model at level N by Shapiro
//                            (InducedRelationsLevel, EquivariantCohomologyOnK);
//   kappa_rank^1             tier A: the rank of kappa^mod_N : H^1(model) -> M_N / Rel_N,
//                            f |-> -f(e_46) ([C] prop:kappamodel (1); = h1 iff an iso);
//   alpha_M=<M>_d=<d>_terms  tier A: the number of block terms of D_{M,N,d}
//                            ([C] eq:degenpres, the admissible pairs, per-eps route);
//   alpha_M=<M>_d=<d>_rank   tier A: the rank of [D_{M,N,d}] = alpha_d between the two
//                            presentations (prop:degenpres (2); OnCokernelDegeneracy);
//   alpha_M=<M>_d=<d>        tier B: that h1_N x h1_M matrix (the H recipe).
// `LOEWY_RING n=<n> ell=<ell> kmax=<K>` (HeckeExt/HeckeRingHeckeAlgebra, the Z/ell^K
//   Hecke-algebra route; [C] cor:stablecert):
//   loewy_ring_blocks        tier A scalar: the number of F_ell-rational systems;
//   loewy_ring_line (<sys>)  tier A text: the LOEWY_RING line of that block byte for byte
//                            (LoewyRingLine: Hilbert functions, K_stab, eps, free rank,
//                            torsion, verdict, stable_cert); every block of the six cells
//                            reads stable_cert=certified (probed before the emission).
// `NEB N=<N> chi=<N.m> k=<k> ell=<ell>` ([C] sec:nebentypus; (NB5) / (NB6) cells of
//   rmk:nebcheckable, chi by its Conrey label, over F_q, q = ell^d):
//   d, h0, h1                tier A scalars of the character model NebCokernel;
//   charpoly_{T,U}_p^1       tier A, p in {2, 3, 5, 7} \ {ell}: the charpoly of
//                            [H_{N,p,chi}] on H^1 (FqCharpoly, low -> high), hashed
//                            FNV(d, #coeffs, then each coefficient's d F_ell-digits);
//                            rows = #coeffs, cols = d.
// Every object is read from a public accessor; nothing re-implements shipped
// mathematics.  MP note: the fixed model is read only through shipped constructors
// (KappaModelPoints locates e_46 by its vertex set); no point sits at ell in {2, 3}.
// ===========================================================================
constexpr uint32_t kLevelNField = 1000003;
struct LevelNAlpha { uint64_t M, d; };
struct LevelNPoint { uint64_t N; uint32_t n; std::vector<LevelNAlpha> alphas; };
const std::vector<LevelNPoint> kGridLevelN = {
    {12, 2, {{2, 6}, {6, 2}}},
    {20, 2, {{10, 2}, {5, 4}}},
    {22, 0, {{11, 2}}},
    {25, 0, {{5, 5}}},
    {30, 0, {{6, 5}, {15, 2}}},
    {35, 2, {{7, 5}, {5, 7}}},
};

uint64_t HashHMatInto(uint64_t h, const hecke::level::HMat& A) {
    h = FnvU64(h, A.rows);
    h = FnvU64(h, A.cols);
    for (uint64_t v : A.a) h = FnvU64(h, v);
    return h;
}
uint32_t HMatRank(const hecke::level::HMat& A, uint32_t p) {
    if (A.rows == 0 || A.cols == 0) return 0;
    std::vector<uint32_t> a(A.a.size());
    for (size_t i = 0; i < a.size(); ++i) a[i] = static_cast<uint32_t>(A.a[i] % p);
    return hecke::level::FpRank(a, A.rows, A.cols, p);
}

void RunLevelNPoint(Collector& C, const SimplicialComplex& K, const LevelNPoint& lp) {
    namespace lv = hecke::level;
    const uint32_t p = kLevelNField;
    const std::string pt = "LEVELN N=" + std::to_string(lp.N) + " n=" + std::to_string(lp.n) + " p=" + std::to_string(p);
    Timer T;
    const engine::ModCtx<uint32_t> ctx(p);
    const lv::Gamma0LevelN lvl(lp.N);
    const lv::SparseCokernel Q = lv::SparseCokernel::Build(lvl, lp.n, ctx, true);
    C.scalar(pt, "h0", 'A', Q.h0());
    C.scalar(pt, "h1", 'A', Q.h1());
    const std::vector<uint32_t> qs = {2, 3, 5, 7};
    const std::map<uint32_t, lv::HMat> H = lv::CokernelHeckeMatrices(lvl, lp.n, ctx, qs);
    for (uint32_t q : qs) {
        const lv::HMat& A = H.at(q);
        const std::string nm = std::string(lp.N % q ? "T_" : "U_") + std::to_string(q);
        const std::vector<uint64_t> cp = lv::HCharpoly(A);
        uint64_t h = FnvU64(kFnvOffset, cp.size());
        for (uint64_t c : cp) h = FnvU64(h, c);
        C.raw(pt, "charpoly_" + nm + "^1", 'A', cp.size(), 1, h);
        C.raw(pt, "H_" + nm + "^1", 'B', A.rows, A.cols, HashHMatInto(kFnvOffset, A));
    }
    {   // kappa^mod_N on the fixed model (prop:kappamodel (1)): HeckeLevelNDegenValidation (K)'s construction
        const lv::KappaPoints pts = lv::KappaModelPoints(K);
        const uint32_t D = lvl.m() * (lp.n + 1);
        const LocalSystemSimplicialComplex L(K, lv::InducedRelationsLevel(K, lvl, lp.n, ctx), D, ctx.p);
        EquivariantCohomologyOnK Hm(&L);
        Hm.compute_cohomology();
        const lv::HFrame f1(&Hm, 1);
        const uint32_t t1 = f1.t();
        C.scalar(pt, "model_dim_H^1", 'A', t1);
        std::vector<uint32_t> co(static_cast<size_t>(Q.h1()) * t1, 0);
        for (uint32_t j = 0; j < t1; ++j) {
            const std::vector<uint32_t> cj = Q.coords_dense(lv::KappaModel(ctx, f1.basis(), t1, j, pts, D));
            for (uint32_t i = 0; i < Q.h1(); ++i) co[static_cast<size_t>(i) * t1 + j] = cj[i];
        }
        C.scalar(pt, "kappa_rank^1", 'A', (t1 && Q.h1()) ? lv::FpRank(co, Q.h1(), t1, p) : 0);
    }
    for (const LevelNAlpha& a : lp.alphas) {
        const std::string nm = "alpha_M=" + std::to_string(a.M) + "_d=" + std::to_string(a.d);
        const lv::Gamma0LevelN lM(a.M);
        const lv::SparseCokernel QM = lv::SparseCokernel::Build(lM, lp.n, ctx, true);
        const lv::DegeneracyOperator Dg(lM, lvl, a.d, lp.n, ctx);
        const lv::HMat A = lv::OnCokernelDegeneracy(QM, Q, Dg);
        C.scalar(pt, nm + "_terms", 'A', Dg.terms().size());
        C.scalar(pt, nm + "_rank", 'A', HMatRank(A, p));
        C.raw(pt, nm, 'B', A.rows, A.cols, HashHMatInto(kFnvOffset, A));
    }
    std::cout << "  [" << pt << "] h1=" << Q.h1() << " " << std::fixed << std::setprecision(2) << T.secs() << " s\n";
}

struct LoewyPoint { uint32_t n; uint64_t ell; uint32_t kmax; };
const std::vector<LoewyPoint> kGridLoewy = {
    {10, 691, 3}, {30, 5, 4}, {34, 5, 3}, {54, 11, 4}, {58, 7, 4}, {82, 7, 4},
};

void RunLoewyPoint(Collector& C, const LoewyPoint& lp) {
    namespace ra = hecke_ext::ringalg;
    const std::string pt = "LOEWY_RING n=" + std::to_string(lp.n) + " ell=" + std::to_string(lp.ell) + " kmax=" +
                           std::to_string(lp.kmax);
    Timer T;
    ra::LoewyOptions o;
    o.kmax = lp.kmax;   // primes: the default {2, 3, 5, 7, ell}; certify: the default (on)
    const ra::LoewyCell c = ra::ComputeRingLoewyCell(lp.n, lp.ell, o);
    C.scalar(pt, "loewy_ring_blocks", 'A', c.blocks.size());
    for (const ra::LoewyBlock& b : c.blocks) {
        std::string sys = "(";
        for (size_t i = 0; i < b.system.size(); ++i) sys += (i ? "," : "") + std::to_string(b.system[i]);
        sys += ")";
        C.text(pt, "loewy_ring_line " + sys, 'A', ra::LoewyRingLine(c, b));
    }
    std::cout << "  [" << pt << "] blocks=" << c.blocks.size() << " " << std::fixed << std::setprecision(2) << T.secs() << " s\n";
}

struct NebPoint { uint64_t N, m; uint32_t k; uint32_t ell; uint32_t d; };
const std::vector<NebPoint> kGridNeb = {
    {4, 3, 5, 13, 1},     // 4.3 = chi_-4, real; an (NB6) CM cell
    {5, 4, 4, 11, 1},     // 5.4 = chi_5, real
    {7, 6, 3, 11, 1},     // 7.6 = chi_-7, real; an (NB6) CM cell
    {7, 2, 4, 13, 1},     // 7.2, order 3, non-real
    {9, 2, 3, 7, 1},      // 9.2, order 6, non-real
    {13, 8, 3, 17, 1},    // 13.8, order 4, non-real
    // (slot-18 WP-4, owner O2 of 2026-10-03; slot-18 R1) the one d = 2 cell: 7.2 (order 3) at
    // ell = 5, where 3 does not divide 5 - 1, so psi lives in F_25 = F_5[x]/(f), f the
    // canonical field polynomial -- pins the F_{ell^d} branch of the recipe below.
    {7, 2, 4, 5, 2},
};

void RunNebPoint(Collector& C, const NebPoint& np) {
    namespace nb = hecke::neb;
    const nb::DirichletCharacter chi = nb::DirichletCharacter::Conrey(np.N, np.m);
    const std::string pt = "NEB N=" + std::to_string(np.N) + " chi=" + chi.label() + " k=" + std::to_string(np.k) +
                           " ell=" + std::to_string(np.ell);
    Timer T;
    const uint32_t n = np.k - 2;
    const nb::Fq F = np.d == 1 ? nb::Fq::Prime(np.ell) : nb::Fq(np.ell, hecke_ext::CanonicalFieldPolynomial(np.ell, np.d));
    const nb::CharacterValues psi = nb::ReduceCharacter(chi, F);
    const hecke::level::Gamma0LevelN lvl(np.N);
    const nb::NebCokernel ck(lvl, n, F, psi);
    ENGINE_CHECK_MSG(!ck.parity_zero(), "golden NEB point: chi(-1) = (-1)^n on the grid");
    C.scalar(pt, "d", 'A', F.degree());
    C.scalar(pt, "h0", 'A', ck.h0());
    C.scalar(pt, "h1", 'A', ck.h1());
    for (uint32_t p : {2u, 3u, 5u, 7u}) {
        if (p == np.ell) continue;
        const nb::HeilbronnChi H(lvl, p, n, F, psi);
        const std::vector<uint32_t> cp = nb::FqCharpoly(F, ck.OnH1(H));
        uint64_t h = FnvU64(kFnvOffset, F.degree());
        h = FnvU64(h, cp.size());
        for (uint32_t c : cp)
            for (uint32_t dig : F.digits(c)) h = FnvU64(h, dig);
        C.raw(pt, std::string("charpoly_") + (np.N % p ? "T_" : "U_") + std::to_string(p) + "^1", 'A', cp.size(),
              F.degree(), h);
    }
    std::cout << "  [" << pt << "] h1=" << ck.h1() << " " << std::fixed << std::setprecision(2) << T.secs() << " s\n";
}

// ===========================================================================
// (hecke-extensions slot 18, WP-4; owner O2 of 2026-10-03 = yes, "the p != 2
// tau / tau_bd tables and one d = 2 NEB cell only") THE ext tau POINTS -- one
// ADD-ONLY emission at NEW points `n=<n> ell=<ell> ext tau` (no pre-existing line
// uses that point string), looped from ComputeAll() ONLY (after the NEB loop),
// never from RunCell32 / RunExtCell32 -- so GoldenSubset32
// (EngineDeterminismValidation), the ext records (HeckeExtDeterminismValidation)
// and the console C4 (a) scripts cannot reach them.  The expect set
// (docs/matrix-engine/rebaseline-2026-10-03-addonly-tau-neb2/expect.tsv) was
// committed before the emission.  The recipe:
//
// On each of the 31 ext-subgrid cells (InExtSubgrid: the cells whose
// `ext cone_action` point carries tau_table_fnv64_p2_r1 / tau_bd_table_fnv64_p2_r1),
// for each (p, nu) of [C, rmk:iotatausym]'s grid other than (2, 1) --
// (3,1), (5,1), (7,1), (2,2), (3,2) -- with p != ell (the cone-action rule: the
// harness never builds a U_ell table, [X] Sec. 1.6; this drops (5,1) at ell = 5):
//   tau_table_fnv64_p<p>_r<nu>     tier B, 1 x 1: HashScalar64 of
//                                  HeckePairHomotopy::tau_table_fnv64() over the
//                                  certified HeckeTauTable of HeckeCosetTables(p, nu,
//                                  ell, n) (alg:tau, obl:tau);
//   tau_bd_table_fnv64_p<p>_r<nu>  tier B, 1 x 1: HashScalar64 of
//                                  HeckePairHomotopy::tau_boundary_table_fnv64()
//                                  (= HeckeBoundaryTauTable::table_fnv64(), checked
//                                  here; lem:taubdryexists, obl:taubdry).
// Exactly the two recipes HeckeConeAction::golden_objects() publishes at p = 2
// (same object names, tier, dims), built on the same cosets / tau / tau_bd / eta
// constructors as RunExtCell32's E5-3 block, under the same u_ell = off scope.
// The certified pair homotopy eta is built (and certified) only because the two
// hashes are its accessors; no eta object is emitted.
// ===========================================================================
struct TauPN { uint64_t p; uint32_t nu; };
const std::vector<TauPN> kGridTauPN = {{3, 1}, {5, 1}, {7, 1}, {2, 2}, {3, 2}};

void RunExtTauCell(Collector& C, const SimplicialComplex& K, const hecke::SL2ZTessellationTables& model,
                   uint32_t n, uint32_t ell) {
    const std::string pt = PointNE(n, ell) + " ext tau";
    Timer T;
    hecke::UEllSetting u_off;
    u_off.mode = hecke::UEllMode::Off;
    const hecke::ScopedUEll u_guard(u_off, hecke::SettingSource::Spec);
    uint32_t tables = 0;
    for (const TauPN& pn : kGridTauPN) {
        if (pn.p == ell) continue;
        const std::string suffix = "_p" + std::to_string(pn.p) + "_r" + std::to_string(pn.nu);
        hecke::HeckeCosetTables cosets(pn.p, pn.nu, ell, n);
        hecke::HeckeTauTable tau(K, model, cosets, ell, n);
        hecke::HeckeBoundaryTauTable taub(K, model, cosets);
        hecke::HeckePairHomotopy eta(tau, taub);
        ENGINE_CHECK_MSG(eta.tau_boundary_table_fnv64() == taub.table_fnv64(),
                         "golden ext tau: the two tau_bd walks disagree");
        C.raw(pt, "tau_table_fnv64" + suffix, 'B', 1, 1, engine::HashScalar64(eta.tau_table_fnv64()));
        C.raw(pt, "tau_bd_table_fnv64" + suffix, 'B', 1, 1, engine::HashScalar64(eta.tau_boundary_table_fnv64()));
        ++tables;
    }
    std::cout << "  [" << pt << "] tables=" << tables << " " << std::fixed << std::setprecision(2) << T.secs() << " s\n";
}

// ===========================================================================
// (hecke-extensions slot 22, G-EVT; the slot-20 candidate list G1 / G2 / G4 / G7,
// docs/implementation/slot20/golden_ledger.md; slot-22 plan Sec. 3 G-EVT, Sec. 5)
// THE LVALUE AND HIDAPAIR POINTS -- one ADD-ONLY emission at NEW points (no
// pre-existing line uses `LVALUE ...` or `HIDAPAIR ...`), looped from ComputeAll()
// ONLY (after the ext tau loop), never from RunCell32 / RunExtCell32 -- so
// GoldenSubset32 (EngineDeterminismValidation), the ext records
// (HeckeExtDeterminismValidation) and the console C4 (a) scripts cannot reach
// them.  The expect set (docs/implementation/slot22/gevt/expect.tsv) and the
// predicted values (docs/implementation/slot22/golden_ledger.md Sec. 3) were
// committed before this code ran.  Every object is tier A (a length, a valuation
// or a sorted multiset of them: basis-independent); every value is read from a
// public accessor of HeckeExt/LValue/ with HeckeLValueValidation's own recipe;
// no expected value is asserted here, only the preconditions under which the
// value exists (ENGINE_CHECK).
//
// `LVALUE n=<n> ell=<ell>` ([E] sec:lvalue):
//   G1 eis_index_m<m>, m = 1, 2, 3   scalar: v_ell(#Q_m) of the congruence route
//                                    EisensteinBernoulliRoute on [C]'s own H^- of the
//                                    7-vertex model ([E] cl:lveis(3); RunEisensteinCell,
//                                    m_max = 3, the (E) arm's recipe);
//   G1 eis_module_c                  scalar: c with [H^+ : H^+_! (+) Z_ell E] = ell^c
//                                    ([E] cl:lveis(2), ReadModuleForm);
//   G2 xkind_congruence              scalar: length T(W_m^{geo-odd}) / I_e on the m_Eis
//                                    block of W = H^1_c(V) (MBlockSign::cong_length);
//   G2 xkind_bilinear                scalar: (H2), v_pi <g_f, h_f> of the cusp form on the
//                                    geo-even (p_0) sign (MBlockForm::mixed);
//   G2 xkind_linear                  scalar: (L1), the depth of r^+ == unit * p_0 read by
//                                    BirchModularSymbolRoute on the group-cocycle W over
//                                    Z/ell^3 (IndexStatement::found);
//   G2 xkind_bernoulli               scalar: v_ell(B_k/k), exact (Bernoulli.h);
//     (G2 is WP-LV3's (LV3-D2) recipe: HidaPairingCell(n, ell, 6, {2,3,5}), the joint
//     m_Eis block in T_2, T_3, T_5, lattice_change on, hd_torsion off.)
//   G7 cf_<plus|minus>_<blk>         raw: FNV(count, the SORTED c_f^eps = lattice_change
//                                    over the forms of the named block on that sign, in
//                                    v_pi units); rows = count, cols = e ([E]
//                                    def:lvlattice, prop:lvlattice; theta = Phi_n).  The
//                                    block: m_Eis (as G2) or, at (258, 37), R19-1's
//                                    m = (37, T_2 - 29) at M = 10 (ArmCell258's recipe).
//                                    Sign plus = geo-even (the p_0 / Eisenstein side).
// `HIDAPAIR k=<k> ell=<ell>` (WP-BIL, R19-9 Sec. 3; [E] rmk:lvshapes (H)):
//   G4 pairing_valuations            raw: FNV(count, then the eigen-systems' tuples
//                                    (e, mixed_even, mixed_odd, hida_defined, hida) SORTED
//                                    lexicographically); rows = count, cols = 5
//                                    (ReadSmallCell(k - 2, ell, MaxPrecision(ell, 6))).
// ===========================================================================
struct LValuePoint {
    uint32_t n;
    uint64_t ell;
    bool g1;            // the congruence route on [C]'s own H^- (cheap cells only)
    bool g2;            // the cross-kind depth on the m_Eis block
    const char* g7;     // nullptr, "mEis" or "T2-29" (the block of c_f^eps)
};
const std::vector<LValuePoint> kGridLValue = {
    {30, 37, true, true, nullptr},
    {10, 691, true, true, nullptr},
    {282, 37, false, true, "mEis"},
    {226, 103, false, true, nullptr},
    {40, 37, false, false, "mEis"},
    {72, 37, false, false, "mEis"},
    {258, 37, false, false, "T2-29"},
};

// The 27 (k, ell) of R19-9 Sec. 3 in WP-BIL's table order (HidaPairingValidationArms.cpp SmallTable()):
// 8 Eisenstein, 6 regular, 3 ramified, 4 split, 5 with n + 1 >= ell, and (12, 23).
const std::vector<std::pair<uint32_t, uint64_t>> kGridHidaPair = {
    {12, 691}, {16, 3617}, {18, 43867}, {20, 283}, {20, 617}, {22, 131}, {22, 593}, {26, 657931},
    {12, 101}, {16, 101}, {18, 101}, {20, 101}, {22, 103}, {26, 101},
    {24, 144169}, {28, 131}, {28, 139},
    {24, 103}, {24, 2294797}, {28, 9349}, {28, 362903},
    {12, 5}, {12, 7}, {12, 11}, {16, 7}, {16, 11},
    {12, 23},
};

// G7: the sorted c_f^eps multiset of one sign of a block reading (v_pi units).
void PushLatticeChange(Collector& C, const std::string& pt, const std::string& obj,
                       const hecke_ext::lvalue::mblock::MBlockSign& S, std::string& line) {
    ENGINE_CHECK_MSG(S.complete, (std::string("golden LVALUE c_f: the m-block reader refused on " + S.name + ": " + S.refusal)).c_str());
    std::vector<uint64_t> cf;
    uint64_t e = 1;
    for (const auto& f : S.forms) {
        ENGINE_CHECK_MSG(f.have_lattice, (std::string("golden LVALUE c_f: the lattice constant was not computed on " + S.name)).c_str());
        ENGINE_CHECK_MSG(f.lattice_change >= 0, "golden LVALUE c_f: c_f < 0");
        ENGINE_CHECK_MSG(cf.empty() || f.e == e, "golden LVALUE c_f: forms of different ramification on one sign");
        e = f.e;
        cf.push_back(static_cast<uint64_t>(f.lattice_change));
    }
    std::sort(cf.begin(), cf.end());
    uint64_t h = FnvU64(kFnvOffset, cf.size());
    for (uint64_t v : cf) h = FnvU64(h, v);
    C.raw(pt, obj, 'A', cf.size(), e, h);
    line += " " + obj + "={";
    for (size_t i = 0; i < cf.size(); ++i) line += (i ? "," : "") + std::to_string(cf[i]);
    line += "}e=" + std::to_string(e);
}

void RunLValuePoint(Collector& C, const LValuePoint& lp) {
    namespace lv = hecke_ext::lvalue;
    namespace hp = hecke_ext::lvalue::hidapair;
    namespace mb = hecke_ext::lvalue::mblock;
    const std::string pt = "LVALUE n=" + std::to_string(lp.n) + " ell=" + std::to_string(lp.ell);
    Timer T;
    std::string line, phases;   // phases: the wall time per sub-reading (printed only; never hashed)
    auto lap = [&phases](const char* what, const Timer& t) {
        std::ostringstream os;
        os << " " << what << "=" << std::fixed << std::setprecision(2) << t.secs() << "s";
        phases += os.str();
    };
    const uint32_t k = lp.n + 2;
    if (lp.g1) {
        const Timer t1;
        lv::EisensteinOptions opt;
        opt.m_max = 3;
        const auto cell = std::make_shared<const lv::EisensteinCell>(lv::RunEisensteinCell(lp.n, lp.ell, opt));
        for (uint32_t m = 1; m <= 3; ++m) {
            const lv::EisensteinBernoulliRoute route(cell, m);
            const uint32_t found = route.index_statement().found;
            C.scalar(pt, "eis_index_m" + std::to_string(m), 'A', found);
            line += " eis_index_m" + std::to_string(m) + "=" + std::to_string(found);
        }
        const lv::ModuleFormReading MF = lv::ReadModuleForm(*cell);
        ENGINE_CHECK_MSG(MF.applicable, (std::string("golden LVALUE G1: the module form cl:lveis(2) is not applicable at " + pt)).c_str());
        C.scalar(pt, "eis_module_c", 'A', MF.c);
        line += " eis_module_c=" + std::to_string(MF.c);
        lap("t_g1", t1);
    }
    if (lp.g2 || (lp.g7 && std::string(lp.g7) == "mEis")) {
        const Timer t2;
        const hp::HidaPairingCell cell(lp.n, lp.ell, 6, {2, 3, 5});
        mb::MBlockOptions opt;
        opt.hd_torsion = false;
        opt.lattice_change = true;
        opt.joint = mb::EisensteinJoint(cell.R(), lp.n);   // m_Eis = (ell, T_p - 1 - p^{k-1} : p = 2, 3, 5)
        const mb::MBlockReading B = mb::ReadMBlockCell(cell, mb::EisensteinT2(cell.R(), lp.n), opt);
        lap("t_cell_mEis", t2);
        if (lp.g2) {
            const mb::MBlockSign& E = B.sign[0];   // geo-even: contains p_0
            const mb::MBlockSign& O = B.sign[1];
            ENGINE_CHECK_MSG(cell.torsion_even().empty() && cell.torsion_odd().empty(),
                             (std::string("golden LVALUE G2: W = H^1_c(V) is not torsion-free at " + pt)).c_str());
            ENGINE_CHECK_MSG(E.complete && O.complete, (std::string("golden LVALUE G2: the m-block reader refused at " + pt)).c_str());
            const mb::MBlockForm* fE = nullptr;
            uint32_t boundary = 0;
            for (const auto& f : E.forms) {
                if (f.eisenstein) ++boundary;
                else fE = &f;
            }
            ENGINE_CHECK_MSG(E.forms.size() == 2 && boundary == 1 && fE != nullptr && O.forms.size() == 1,
                             (std::string("golden LVALUE G2: the m_Eis block is not <boundary line, f> (+) <f> at " + pt)).c_str());
            // the linear kind: (L1) on <p_0, r^+>, BirchModularSymbolRoute on the group-cocycle W over Z/ell^3
            uint32_t depth = 0;
            const Timer t3;
            {
                lv::birch::BirchConeCell bc;
                bc.n = lp.n;
                bc.ell = lp.ell;
                bc.m = 3;
                bc.R = lv::birch::MakeRing(lp.ell, 3);
                for (const hp::Vec& w : cell.W()) {
                    lv::birch::Vec p(w.size());
                    for (size_t i = 0; i < w.size(); ++i) p[i] = w[i] % bc.R.q;
                    bc.P.push_back(p);
                    bc.exps.push_back(3);
                }
                bc.route = "group-cocycle W = ker(1+S) cap ker(1+U+U^2) (hidapair::HidaPairingCell), not the C++ cone";
                const auto bcell = std::make_shared<const lv::birch::BirchConeCell>(bc);
                const lv::BirchModularSymbolRoute route(bcell, lv::birch::TrivialCharacter(bc.R), 0);
                depth = route.index_statement().found;
            }
            lap("t_linear", t3);
            const Timer t4;
            const lv::BernoulliReading Bk = lv::BernoulliModEllPowerExact(k, lp.ell, 3);
            lap("t_bernoulli", t4);
            ENGINE_CHECK_MSG(Bk.v_exact, "golden LVALUE G2: v(B_k/k) is not exact at precision 3");
            C.scalar(pt, "xkind_congruence", 'A', O.cong_length);
            C.scalar(pt, "xkind_bilinear", 'A', fE->mixed);
            C.scalar(pt, "xkind_linear", 'A', depth);
            C.scalar(pt, "xkind_bernoulli", 'A', Bk.v);
            line += " xkind_congruence=" + std::to_string(O.cong_length) + " xkind_bilinear=" + std::to_string(fE->mixed) +
                    " xkind_linear=" + std::to_string(depth) + " xkind_bernoulli=" + std::to_string(Bk.v);
        }
        if (lp.g7 && std::string(lp.g7) == "mEis") {
            PushLatticeChange(C, pt, "cf_plus_mEis", B.sign[0], line);
            PushLatticeChange(C, pt, "cf_minus_mEis", B.sign[1], line);
        }
    }
    if (lp.g7 && std::string(lp.g7) == "T2-29") {
        ENGINE_CHECK_MSG(lp.n == 258 && lp.ell == 37, "golden LVALUE: the block m = (37, T_2 - 29) is R19-1's (258, 37) only");
        const Timer t5;
        const hp::HidaPairingCell cell(lp.n, lp.ell, 10, {2, 3, 5});
        mb::MBlockOptions opt;
        opt.hd_torsion = false;
        opt.lattice_change = true;
        const mb::MBlockReading B = mb::ReadMBlockCell(cell, 29, opt);
        PushLatticeChange(C, pt, "cf_plus_T2-29", B.sign[0], line);
        PushLatticeChange(C, pt, "cf_minus_T2-29", B.sign[1], line);
        lap("t_cell_T2-29", t5);
    }
    std::cout << "  [" << pt << "]" << line << " |" << phases << " " << std::fixed << std::setprecision(2) << T.secs() << " s\n";
}

void RunHidaPairPoint(Collector& C, uint32_t k, uint64_t ell) {
    namespace hp = hecke_ext::lvalue::hidapair;
    const std::string pt = "HIDAPAIR k=" + std::to_string(k) + " ell=" + std::to_string(ell);
    Timer T;
    const hp::SmallCellReading r = hp::ReadSmallCell(k - 2, ell, hp::Zq::MaxPrecision(ell, 6));
    std::vector<std::vector<uint64_t>> tuples;
    for (const hp::EigenReading& E : r.systems)
        tuples.push_back({E.e, E.mixed_even, E.mixed_odd, E.hida_defined ? 1u : 0u, E.hida_defined ? E.hida : 0u});
    std::sort(tuples.begin(), tuples.end());
    uint64_t h = FnvU64(kFnvOffset, tuples.size());
    std::string line;
    for (const auto& t : tuples) {
        line += " (";
        for (size_t i = 0; i < t.size(); ++i) {
            h = FnvU64(h, t[i]);
            line += (i ? "," : "") + std::to_string(t[i]);
        }
        line += ")";
    }
    C.raw(pt, "pairing_valuations", 'A', tuples.size(), 5, h);
    std::cout << "  [" << pt << "] kind=" << r.kind << line << " " << std::fixed << std::setprecision(2) << T.secs() << " s\n";
}

// ===========================================================================
// (slot-24 D6, DESCENT_plan.md Sec. 4 D6; [X] OD-9) THE DESCENT POINTS -- the
// integral descent record integral_descent_record/v1 (Integral/Descent/
// DescentRecord.h; [C] thm:descentcert(1)-(3), sec:descentpair) at nine
// cheap cells.  STAGED, NOT EMITTED: RunDescentPoints is reached from
// GoldenDescentRecords() (IntegralDescentRecordValidation's arm (G)) only;
// ComputeAll() calls it only in the owner-gated landing commit of the ONE
// add-only event (docs/implementation/slot24/d6golden/expect.tsv, the
// prediction), after the HIDAPAIR loop, so every new line lies after every
// pre-existing data line and no existing point, object or loop order moves.
//
// Point string `DESCENT N=<N> n=<n>` (new; used by no pre-existing line).  The
// cell's Hecke primes are the record's default P (the first three of 2, 3, 5,
// 7, 11, 13 not dividing N).  coefficient_ring = Z is installed for the record
// only (ScopedCoefficientRing(0), source spec; the E8 precedent).  Every
// object is TIER A: an isomorphism class of a finitely generated abelian group
// or of a finite Z[T_p]-module, or a characteristic polynomial over Z -- each
// independent of every choice by thm:descentcert's independence clause.  A
// line exists only for a record with status ok, every char_Z determined and
// every torsion module certified with commuting T_p (ENGINE_CHECK).
//
//   H0 .. H3                text: AbelianGroupType::to_string() of H^q(Gamma_0(N); Sym^n Z^2)
//   bad_primes              text: "ell:count,..." of prop:badprimesN(1) ("" when B is empty)
//   charZ_T<p>              raw: rows = r + 1 (the coefficients), cols = the undetermined count (0),
//                           hash = D3's CharpolyZ::digest() (r, then per i: i, 1 + sign, c_i mod 2^64,
//                           c_i mod 3^40) -- choice-free on certified coefficients
//   tors_q<q>_ell<ell>      text: "a=<exponents ascending, '.'-joined>" then " T_<p>:<canonical invariants>"
//                           per p (D3's canonical text, prop:torsionhecke(1)); every (q, ell) the record
//                           reads, an empty group included
//   pair_H, pair_Hd, pair_Hc  text: the groups q = 0 .. 4 of the boundary pair, " | "-joined
//
// The landing commit dates the next line (golden_landing_check.sh needs the
// harness and a rebaseline note in it); it is a comment, outside
// kProvenanceNote, so the emitted reference does not depend on it.
// LANDING: 2026-10-26, the add-only event R1 (docs/implementation/slot24/d6golden/ledger.md Sec. 7)
// ===========================================================================
struct DescentPoint {
    uint32_t N, n;
};
const std::vector<DescentPoint> kGridDescent = {
    {1, 10}, {1, 12}, {1, 18}, {1, 3}, {2, 8}, {3, 4}, {5, 4}, {11, 0}, {11, 2},
};

void RunDescentPoint(Collector& C, const DescentPoint& dp) {
    namespace rec = integral_descent::record;
    const std::string pt = "DESCENT N=" + std::to_string(dp.N) + " n=" + std::to_string(dp.n);
    Timer T;
    const hecke::ScopedCoefficientRing ring(0, hecke::SettingSource::Spec);
    const rec::DescentRecordZ R = rec::BuildDescentRecord(dp.N, dp.n);
    ENGINE_CHECK_MSG(R.ok(), (std::string("golden DESCENT: the record is refused at ") + pt + ": " + R.refusal).c_str());
    ENGINE_CHECK_MSG(R.H.size() == rec::kRecordQMax + 1, "golden DESCENT: the record does not report H^0 .. H^q_max");
    std::string line;
    for (uint32_t q = 0; q <= rec::kRecordQMax; ++q) {
        C.text(pt, "H" + std::to_string(q), 'A', R.H[q].to_string());
        line += " H" + std::to_string(q) + "=" + R.H[q].to_string();
    }
    std::string bad;
    for (size_t i = 0; i < R.bad_primes.size(); ++i)
        bad += (i ? "," : "") + std::to_string(R.bad_primes[i].first) + ":" + std::to_string(R.bad_primes[i].second);
    C.text(pt, "bad_primes", 'A', bad);
    for (const rec::CharpolyEntry& c : R.charpoly) {
        ENGINE_CHECK_MSG(c.determined && !c.refused, (std::string("golden DESCENT: char_Z(T_p) not determined at ") + pt).c_str());
        C.raw(pt, "charZ_T" + std::to_string(c.p), 'A', c.r + 1, c.undetermined, c.digest);
        line += " charZ_T" + std::to_string(c.p) + "=";
        for (size_t i = 0; i < c.coefficients.size(); ++i) line += (i ? "," : "") + c.coefficients[i];
    }
    for (const rec::TorsionModule& m : R.torsion) {
        ENGINE_CHECK_MSG(!m.refused, (std::string("golden DESCENT: a torsion read is refused at ") + pt).c_str());
        // (D6 audit N7) tier A holds certified modules only: module_certified (2 sigma_max <= K, so the canonical text is the full one)
        // and commuting T_p -- neither is in the text, so an uncertified read would otherwise pass as a shorter canonical text
        ENGINE_CHECK_MSG(m.module_certified && m.commute,
                         (std::string("golden DESCENT: a torsion module is not certified / its T_p do not commute at ") + pt + " (q, ell) = (" +
                          std::to_string(m.q) + ", " + std::to_string(m.ell) + ")")
                             .c_str());
        std::string s = "a=";
        for (size_t i = 0; i < m.exponents.size(); ++i) s += (i ? "." : "") + std::to_string(m.exponents[i]);
        for (const auto& [p, canon] : m.invariants) s += " T_" + std::to_string(p) + ":" + canon;
        C.text(pt, "tors_q" + std::to_string(m.q) + "_ell" + std::to_string(m.ell), 'A', s);
    }
    ENGINE_CHECK_MSG(R.pair.ran && !R.pair.refused, (std::string("golden DESCENT: the pair is refused at ") + pt).c_str());
    auto joined = [](const std::vector<integral_model::AbelianGroupType>& G) {
        std::string s;
        for (size_t q = 0; q < G.size(); ++q) s += (q ? " | " : "") + G[q].to_string();
        return s;
    };
    C.text(pt, "pair_H", 'A', joined(R.pair.H));
    C.text(pt, "pair_Hd", 'A', joined(R.pair.Hd));
    C.text(pt, "pair_Hc", 'A', joined(R.pair.Hc));
    std::cout << "  [" << pt << "]" << line << " bad=" << (bad.empty() ? "-" : bad) << " " << std::fixed << std::setprecision(2) << T.secs()
              << " s\n";
}

void RunDescentPoints(Collector& C) {
    for (const DescentPoint& dp : kGridDescent) RunDescentPoint(C, dp);
}

// ===========================================================================
// (hecke-extensions slot 23, G-EVT23; docs/implementation/slot23/G-EVT23_plan.md, expect set
// docs/implementation/slot23/gevt23/expect23.tsv, note docs/matrix-engine/rebaseline-2026-10-13-addonly-slot22-families.md)
// THE TAMEFAM, PRANK, TOWER, SHAPIRO AND LESLEVEL POINTS -- one ADD-ONLY emission of the slot-22 families WAKE, NF-PRANK,
// P10, P6 and LES-1 at NEW points (no pre-existing line uses any of the five point strings), looped from ComputeAll()
// ONLY (after the DESCENT loop -- slot-25 GOLD-PORT; the archived slot-23 branch, written before D6, put them after the
// HIDAPAIR loop), never from RunCell32 / RunExtCell32 / the console scripts -- so GoldenSubset32
// (EngineDeterminismValidation), the ext records (HeckeExtDeterminismValidation) and the console C4 (a) scripts cannot
// reach them.  Every object is tier A (an integer, a tuple of integers or a sorted multiset of them: basis-, root-order-
// and normalisation-independent); every value is read from the public accessor the family's validation reads, with the
// validation's own set-up (HpwweSpec, DefaultTowerOptions, ShapiroEigenLine, the les:: tables -- moved out of the
// validations for this, value-neutral); no expected value is asserted here, only the preconditions under which the
// value exists (ENGINE_CHECK).  Analytic quantities (hR, R', Euler products) are not recorded.
//
// THE GATE: kGEvt23InReference -- TRUE since the HZ-GOLD landing (slot 25, owner go 2026-11-07; docs/implementation/slot25/
// golden_ledger.md, docs/matrix-engine/rebaseline-s25-hzgold-addonly-gevt23-heckez.md): every caller computes the five
// families and GoldenRegressionValidation checks the 97 entries like every other.  (Before the landing it was false and the
// families were computed only under the scratch interlock GOLDEN_GEVT23=1 into a GOLDEN_OUT outside the tree; the landing
// RETIRED that interlock -- the variable is no longer read.)  The gate selects WHICH records are computed, never a value;
// it is tied to kHZInReference by a static_assert (one event).
//
// Raw objects hash FNV-1a-64 (Fnv.h) of: the element count, then the elements in the stated order (HashSeq); a
// decimal string (d_K, the index, an invariant factor) enters by its bytes (FnvBytes).  rows / cols as stated.
//
// `TAMEFAM p=5 l0=11 l1=23` (WAKE V1 / V2; l1 = 67 dropped for cost, see kGridTameFam; RunTameCell on HpwweSpec(l1), controls / wild / identities off -- they
//   feed no recorded object):  scalars tame_m, tame_h, tame_c, tame_cusp_m, tame_full_m, tame_dimA, tame_full_alg_dim;
//   raw tame_ideal_dims = (dim A/(y), dim A/(y+z), dim A/(y-z)), rows 1, cols 3;
//   raw tame_red = (dim A/J^red, surjective, J = (y), J = (y+z), J = (y-z)), rows 1, cols 5.
// `PRANK N=<N> p=5` (NF-PRANK; nf_prank, analytic side off -- it feeds no recorded object; B_0 = M_K at the gate cells
//   11, 31, 41, 61 (G3's recipe); the CE cells 211, 401, 691 of the plan are dropped for cost, see kGridPRank):
//   raw prank_field = (d_K, r1, r2, [O_K : Z[alpha]]), rows 1, cols 4; raw prank_chev = (t, lambda ramified, r*,
//   Chevalley exponent), rows 1, cols 4; scalars prank_rank_upper, prank_rank_lower; raw prank_group_bound = the
//   invariant factors != 1 of Z^{S_0}/R_0, sorted ascending, rows = count, cols 1.
// `PRANK chevalley p<=97` (NF-PRANK G2): raw chevalley_table = per cell of chevalley_cells.tsv (sha256-checked, file
//   order) (N, p, t, lambda ramified, r*, exponent) of chevalley_data(N, p), rows = cells (1240), cols 6.
// `TOWER ell=<l> a=<a> r=<r>` (P10; BuildTower(cell, DefaultTowerOptions(cell)), K = 3): scalars tower_m, tower_h,
//   tower_rank; raw tower_eigs = the sorted residual eigenvalues of T_2 on E, then those of T_3, rows = rank, cols 2.
// `SHAPIRO N=<N> n=<n> ell=<l>` (P6; ShapiroEigenLine(N, n, ell, 3), sign -1): raw shapiro_ap = the pairs (p, a_p mod
//   ell^3), p ascending over ShapiroLinePrimes(N), rows 5, cols 2; scalar shapiro_rankW.
// `LESLEVEL` (LES-1; the les:: tables of HeckeLevelPair.h): scalars les_g0_pairs, les_g0_normal, les_idx_principal_M<M>
//   (M = 2, 3, 4, 5, 7: [Gamma : Gamma(M)]), les_idx_comm2_cap_comm3 ([Gamma : Gamma^2 cap Gamma^3]); raw
//   les_g0_normal_list = the normal (N, M) in table order, rows = count, cols 2; raw les_g1_steps = (pairs, #(a), #(b),
//   #(c)), rows 1, cols 4; raw les_model_tags = per row of ModelTagTable (name, -I in, order-3, tag at 2, 3, 5),
//   rows 8, cols 5.
// ===========================================================================
constexpr bool kGEvt23InReference = true;   // HZ-GOLD landing, 2026-11-07

uint64_t HashSeq(const std::vector<uint64_t>& v) {
    uint64_t h = FnvU64(kFnvOffset, v.size());
    for (uint64_t x : v) h = FnvU64(h, x);
    return h;
}

std::string HexFnv(uint64_t v) {
    std::ostringstream os;
    os << "0x" << std::hex << std::setw(16) << std::setfill('0') << v;
    return os.str();
}

std::string JoinU64(const std::vector<uint64_t>& v) {
    std::string s;
    for (size_t i = 0; i < v.size(); ++i) s += (i ? "," : "") + std::to_string(v[i]);
    return s;
}

// (cost, G-EVT23 plan Sec. 6 drop order step 2) l1 = 67 (h = 681; 4.9 s of the 14.6 s measured for the full plan set) is
// NOT recorded; it stays in HeckeTameFamilyValidation arm (H).  Restoring it = {23, 67} + gen_expect23.py, before the emission.
const std::vector<uint64_t> kGridTameFam = {23};

void RunTameFamPoint(Collector& C, uint64_t l1) {
    namespace tm = hecke::tame;
    const std::string pt = "TAMEFAM p=5 l0=11 l1=" + std::to_string(l1);
    Timer T;
    tm::TameCellSpec spec = tm::HpwweSpec(l1);
    ENGINE_CHECK_MSG(spec.pres != nullptr && spec.p == 5 && spec.l0 == 11, "golden TAMEFAM: not an HPWWE cell");
    spec.controls = false;
    spec.wild = false;
    spec.identities = false;
    const tm::TameCellReading r = tm::RunTameCell(spec);
    ENGINE_CHECK_MSG(r.genus_ok && r.rel_boundary_ok && r.boundary_rank + 1 == r.c && r.rel_escapes == 0,
                     (std::string("golden TAMEFAM: the quotient model fails its structure checks at " + pt)).c_str());
    ENGINE_CHECK_MSG(r.pres_done && r.v2_done(), (std::string("golden TAMEFAM: presentation / V2 not computed at " + pt)).c_str());
    C.scalar(pt, "tame_m", 'A', r.m);
    C.scalar(pt, "tame_h", 'A', r.h);
    C.scalar(pt, "tame_c", 'A', r.c);
    C.scalar(pt, "tame_cusp_m", 'A', r.cusp_m);
    C.scalar(pt, "tame_full_m", 'A', r.full_m);
    C.scalar(pt, "tame_dimA", 'A', r.dimA);
    C.scalar(pt, "tame_full_alg_dim", 'A', r.full_alg_dim);
    const std::vector<uint64_t> ideal = {r.dim_y, r.dim_ypz, r.dim_ymz};
    C.raw(pt, "tame_ideal_dims", 'A', 1, 3, HashSeq(ideal));
    const std::vector<uint64_t> red = {r.red.dim_quotient, r.red.surjective ? 1u : 0u, r.J_is_y ? 1u : 0u, r.J_is_ypz ? 1u : 0u,
                                       r.J_is_ymz ? 1u : 0u};
    C.raw(pt, "tame_red", 'A', 1, 5, HashSeq(red));
    std::cout << "  [" << pt << "] m=" << r.m << " h=" << r.h << " c=" << r.c << " cusp_m=" << r.cusp_m << " full_m=" << r.full_m
              << " dimA=" << r.dimA << " full_alg_dim=" << r.full_alg_dim << " ideal_dims=(" << JoinU64(ideal) << ") red=("
              << JoinU64(red) << ") case=" << r.red.case_name() << " " << std::fixed << std::setprecision(2) << T.secs() << " s\n";
}

struct PRankPoint {
    uint32_t N;
    uint64_t dense_bound;   // 0 = M_K (G3's recipe); 20000 = the NF_PRANK_CE hook's default
};
// (cost, G-EVT23 plan Sec. 6 drop order step 1) the CE cells {211, 20000}, {401, 20000}, {691, 20000} (5.6 s of the 14.6 s
// measured for the full plan set; 691 alone 3.9 s) are NOT recorded; they stay in the read-only NF_PRANK_CE hook.
const std::vector<PRankPoint> kGridPRank = {{11, 0}, {31, 0}, {41, 0}, {61, 0}};

void RunPRankPoint(Collector& C, const PRankPoint& pp) {
    const std::string pt = "PRANK N=" + std::to_string(pp.N) + " p=5";
    Timer T;
    nf::PRankOptions o;
    o.dense_bound = pp.dense_bound;
    o.analytic = false;   // R', the Euler product and the index certificate feed no recorded object
    const nf::PRankResult R = nf::nf_prank(pp.N, 5, o);
    ENGINE_CHECK_MSG(R.level != nf::PRankLevel::REFUSED && R.minkowski_complete && R.genus_ok && R.order_maximal,
                     (std::string("golden PRANK: the class-group bound is not certified at " + pt + ": " + R.refusal)).c_str());
    ENGINE_CHECK_MSG(R.chev.refusal.empty(), (std::string("golden PRANK: Chevalley data refused at " + pt)).c_str());
    const std::string dK = R.dK.to_string(), idx = R.index.to_string();
    uint64_t hf = FnvU64(kFnvOffset, 4);
    hf = FnvBytes(hf, dK);
    hf = FnvU64(hf, R.r1);
    hf = FnvU64(hf, R.r2);
    hf = FnvBytes(hf, idx);
    C.raw(pt, "prank_field", 'A', 1, 4, hf);
    const std::vector<uint64_t> chev = {R.chev.t, R.chev.lambda_ramified ? 1u : 0u, R.chev.rstar, R.chev.chev_exponent};
    C.raw(pt, "prank_chev", 'A', 1, 4, HashSeq(chev));
    C.scalar(pt, "prank_rank_upper", 'A', R.rank_upper);
    C.scalar(pt, "prank_rank_lower", 'A', R.rank_lower);
    std::vector<std::string> gb;
    for (const nf::BigInt& d : R.group_bound) {
        ENGINE_CHECK_MSG(!d.is_zero(), "golden PRANK: Z^{S_0}/R_0 has a free part");
        gb.push_back(d.to_string());
    }
    std::sort(gb.begin(), gb.end(), [](const std::string& a, const std::string& b) {
        return a.size() != b.size() ? a.size() < b.size() : a < b;   // numeric order of non-negative decimals
    });
    uint64_t hg = FnvU64(kFnvOffset, gb.size());
    std::string gbs;
    for (size_t i = 0; i < gb.size(); ++i) {
        hg = FnvBytes(hg, gb[i]);
        gbs += (i ? "," : "") + gb[i];
    }
    C.raw(pt, "prank_group_bound", 'A', gb.size(), 1, hg);
    std::cout << "  [" << pt << "] dK=" << dK << " sig=(" << R.r1 << "," << R.r2 << ") index=" << idx << " chev=(" << JoinU64(chev)
              << ") rank_upper=" << R.rank_upper << " rank_lower=" << R.rank_lower << " group_bound=[" << gbs << "] B0=" << R.dense_bound
              << " level=" << nf::prank_level_name(R.level) << " " << std::fixed << std::setprecision(2) << T.secs() << " s\n";
}

std::string SourceRoot() {
#ifdef OC_SOURCE_DIR
    return std::string(OC_SOURCE_DIR);
#else
    std::string f = __FILE__;   // <root>/src/hacs/Golden/GoldenHarness.cpp
    for (int k = 0; k < 4; ++k) {
        const size_t slash = f.find_last_of("/\\");
        f = slash == std::string::npos ? std::string(".") : f.substr(0, slash);
    }
    return f;
#endif
}

void RunChevalleyTablePoint(Collector& C) {
    const std::string pt = "PRANK chevalley p<=97";
    Timer T;
    const std::string path = SourceRoot() + "/docs/implementation/slot22/nfprank/chevalley_cells.tsv";
    std::ifstream in(path, std::ios::binary);
    ENGINE_CHECK_MSG(static_cast<bool>(in), (std::string("golden PRANK chevalley: cannot read " + path)).c_str());
    std::stringstream buf;
    buf << in.rdbuf();
    const std::string text = buf.str();
    nf::Sha256 sh;
    sh.update(text.data(), text.size());
    ENGINE_CHECK_MSG(sh.hex() == "3d6f6903f8ae47da017909be2ccca6a1ea961243d4471fb155fc40de71f22154",
                     "golden PRANK chevalley: chevalley_cells.tsv is not the frozen oracle (sha256)");
    std::istringstream ls(text);
    std::string line;
    std::vector<std::vector<uint64_t>> cells;
    while (std::getline(ls, line)) {
        if (line.empty() || line[0] == '#' || line[0] == 'N') continue;
        std::stringstream fs(line);
        std::string a, b;
        std::getline(fs, a, '\t');
        std::getline(fs, b, '\t');
        const uint32_t N = static_cast<uint32_t>(std::stoul(a)), p = static_cast<uint32_t>(std::stoul(b));
        const nf::ChevalleyData D = nf::chevalley_data(N, p);
        ENGINE_CHECK_MSG(D.refusal.empty(), (std::string("golden PRANK chevalley: refused at N=" + a + " p=" + b)).c_str());
        cells.push_back({N, p, D.t, D.lambda_ramified ? 1u : 0u, D.rstar, D.chev_exponent});
    }
    uint64_t h = FnvU64(kFnvOffset, cells.size());
    for (const auto& c : cells)
        for (uint64_t x : c) h = FnvU64(h, x);
    C.raw(pt, "chevalley_table", 'A', cells.size(), 6, h);
    std::cout << "  [" << pt << "] cells=" << cells.size() << " fnv=" << HexFnv(h) << " " << std::fixed << std::setprecision(2) << T.secs()
              << " s\n";
}

const std::vector<hecke_ext::tower::TowerCell> kGridTower = {
    {5, 0, 1, 3}, {5, 0, 2, 3}, {5, 2, 1, 3}, {5, 2, 2, 3}, {7, 0, 1, 3},
    {7, 0, 2, 3}, {7, 2, 1, 3}, {7, 2, 2, 3}, {11, 2, 1, 3}, {13, 0, 1, 3},
};

void RunTowerPoint(Collector& C, const hecke_ext::tower::TowerCell& cell) {
    namespace tw = hecke_ext::tower;
    const std::string pt = "TOWER ell=" + std::to_string(cell.ell) + " a=" + std::to_string(cell.a) + " r=" + std::to_string(cell.r);
    Timer T;
    const tw::TowerReading t = tw::BuildTower(cell, tw::DefaultTowerOptions(cell));
    uint32_t esc = 0;
    for (const auto& kv : t.rel_escapes) esc += kv.second;
    ENGINE_CHECK_MSG(esc == 0 && t.krylov_stable && t.E_free && t.E_Ustable && t.U_invertible && t.split_ok &&
                         (cell.r > 2 || (t.g5prime_run && t.g5prime_ok)),
                     (std::string("golden TOWER: the ordinary part is not certified at " + pt)).c_str());
    ENGINE_CHECK_MSG(t.eigsE.count("T2") && t.eigsE.count("T3"), (std::string("golden TOWER: no residual eigenvalues at " + pt)).c_str());
    C.scalar(pt, "tower_m", 'A', t.m);
    C.scalar(pt, "tower_h", 'A', t.h);
    C.scalar(pt, "tower_rank", 'A', t.rank);
    std::vector<uint64_t> e2(t.eigsE.at("T2").begin(), t.eigsE.at("T2").end()), e3(t.eigsE.at("T3").begin(), t.eigsE.at("T3").end());
    std::sort(e2.begin(), e2.end());
    std::sort(e3.begin(), e3.end());
    ENGINE_CHECK_MSG(e2.size() == t.rank && e3.size() == t.rank, "golden TOWER: residual eigenvalue count != rank");
    uint64_t h = FnvU64(kFnvOffset, e2.size());
    for (uint64_t x : e2) h = FnvU64(h, x);
    h = FnvU64(h, e3.size());
    for (uint64_t x : e3) h = FnvU64(h, x);
    C.raw(pt, "tower_eigs", 'A', t.rank, 2, h);
    std::cout << "  [" << pt << "] m=" << t.m << " h=" << t.h << " rank=" << t.rank << " eigs_T2={" << JoinU64(e2) << "} eigs_T3={"
              << JoinU64(e3) << "} " << std::fixed << std::setprecision(2) << T.secs() << " s\n";
}

struct ShapiroPoint {
    uint64_t N;
    uint32_t n;
    uint64_t ell;
};
const std::vector<ShapiroPoint> kGridShapiro = {{2, 6, 13}, {3, 4, 13}, {5, 2, 13}, {7, 2, 13},
                                                {11, 0, 13}, {11, 0, 11}, {11, 0, 5}, {11, 0, 7}};

void RunShapiroPoint(Collector& C, const ShapiroPoint& sp) {
    namespace sh = hecke_ext::lvalue::shapiro;
    const std::string pt = "SHAPIRO N=" + std::to_string(sp.N) + " n=" + std::to_string(sp.n) + " ell=" + std::to_string(sp.ell);
    Timer T;
    const sh::ShapiroEigenLineReading r = sh::ShapiroEigenLine(sp.N, sp.n, sp.ell, 3);
    ENGINE_CHECK_MSG(r.rank_one && r.all_eigen, (std::string("golden SHAPIRO: W^- is not an eigen-line at " + pt)).c_str());
    uint64_t h = FnvU64(kFnvOffset, r.ap.size());
    std::string line;
    for (const auto& pa : r.ap) {
        h = FnvU64(h, pa.first);
        h = FnvU64(h, pa.second);
        line += " a_" + std::to_string(pa.first) + "=" + std::to_string(pa.second);
    }
    C.raw(pt, "shapiro_ap", 'A', r.ap.size(), 2, h);
    C.scalar(pt, "shapiro_rankW", 'A', r.rankW);
    std::cout << "  [" << pt << "]" << line << " rankW=" << r.rankW << " " << std::fixed << std::setprecision(2) << T.secs() << " s\n";
}

void RunLesLevelPoint(Collector& C) {
    namespace les = hecke::level::les;
    const std::string pt = "LESLEVEL";
    Timer T;
    std::vector<uint64_t> normal_list;
    uint64_t pairs = 0, normal = 0;
    for (const les::Gamma0NormalRow& r : les::Gamma0NormalityTable()) {
        ENGINE_CHECK_MSG(r.contained, "golden LESLEVEL: Gamma_0(MN) is not contained in Gamma_0(N)");
        ++pairs;
        if (r.normal) {
            ++normal;
            normal_list.push_back(r.N);
            normal_list.push_back(r.M);
        }
    }
    C.scalar(pt, "les_g0_pairs", 'A', pairs);
    C.scalar(pt, "les_g0_normal", 'A', normal);
    uint64_t hn = FnvU64(kFnvOffset, normal);
    for (uint64_t x : normal_list) hn = FnvU64(hn, x);
    C.raw(pt, "les_g0_normal_list", 'A', normal, 2, hn);
    std::string idx;
    for (uint64_t M : {2ull, 3ull, 4ull, 5ull, 7ull}) {
        const les::LevelHandle GM = les::MakeLevel(std::make_unique<les::CongruenceLevel>(les::CongruenceLevel::Principal(M)));
        C.scalar(pt, "les_idx_principal_M" + std::to_string(M), 'A', GM.perm.m);
        idx += " M" + std::to_string(M) + "=" + std::to_string(GM.perm.m);
    }
    const les::CosetPermutation p2 = les::PermutationOf(les::CongruenceLevel::Commutator(2));
    const les::CosetPermutation p3 = les::PermutationOf(les::CongruenceLevel::Commutator(3));
    const uint64_t i23 = les::IntersectionIndex(p2, p3);
    C.scalar(pt, "les_idx_comm2_cap_comm3", 'A', i23);
    std::vector<uint64_t> steps(4, 0);
    for (const les::Gamma1StepRow& r : les::Gamma1StepsTable()) {
        ++steps[0];
        steps[1] += r.a;
        steps[2] += r.b;
        steps[3] += r.c;
    }
    C.raw(pt, "les_g1_steps", 'A', 1, 4, HashSeq(steps));
    const std::vector<les::ModelTagRow> tags = les::ModelTagTable();
    uint64_t ht = FnvU64(kFnvOffset, tags.size());
    std::string tl;
    for (const les::ModelTagRow& r : tags) {
        ht = FnvBytes(ht, r.name);
        ht = FnvU64(ht, r.t.minus_I ? 1u : 0u);
        ht = FnvU64(ht, r.t.order3 ? 1u : 0u);
        ht = FnvU64(ht, static_cast<uint64_t>(r.t2));
        ht = FnvU64(ht, static_cast<uint64_t>(r.t3));
        ht = FnvU64(ht, static_cast<uint64_t>(r.t5));
        tl += " " + r.name + ":" + (r.t.minus_I ? "1" : "0") + (r.t.order3 ? "1" : "0") + "/" + les::ModelTagName(r.t2) + "," +
              les::ModelTagName(r.t3) + "," + les::ModelTagName(r.t5);
    }
    C.raw(pt, "les_model_tags", 'A', tags.size(), 5, ht);
    std::cout << "  [" << pt << "] g0_pairs=" << pairs << " g0_normal=" << normal << " normal_fnv=" << HexFnv(hn) << idx
              << " comm2_cap_comm3=" << i23 << " g1_steps=(" << JoinU64(steps) << ") tags=" << tl << " " << std::fixed
              << std::setprecision(2) << T.secs() << " s\n";
}

// The five G-EVT23 families, in the emission order of the expect set (one banner each; per-family wall time printed).
void RunGEvt23Points(Collector& C) {
    Timer T;
    std::cout << "--- TAMEFAM points (WAKE: the HPWWE tame-family cells) ---\n";
    for (uint64_t l1 : kGridTameFam) RunTameFamPoint(C, l1);
    const double t_wake = T.secs();
    std::cout << "--- PRANK points (NF-PRANK: 5-ranks of Cl(Q(N^{1/5})), the Chevalley table) ---\n";
    for (const PRankPoint& pp : kGridPRank) RunPRankPoint(C, pp);
    RunChevalleyTablePoint(C);
    const double t_prank = T.secs();
    std::cout << "--- TOWER points (P10: the Lambda-adic Hida tower, r <= 2) ---\n";
    for (const auto& cell : kGridTower) RunTowerPoint(C, cell);
    const double t_tower = T.secs();
    std::cout << "--- SHAPIRO points (P6: Gamma_0(N) eigen-lines via Shapiro) ---\n";
    for (const ShapiroPoint& sp : kGridShapiro) RunShapiroPoint(C, sp);
    const double t_shapiro = T.secs();
    std::cout << "--- LESLEVEL point (LES-1: level-change integers) ---\n";
    RunLesLevelPoint(C);
    const double t_les = T.secs();
    std::cout << "  G-EVT23 family walls: WAKE=" << std::fixed << std::setprecision(2) << t_wake << "s NFPRANK=" << (t_prank - t_wake)
              << "s P10=" << (t_tower - t_prank) << "s P6=" << (t_shapiro - t_tower) << "s LES1=" << (t_les - t_shapiro) << "s total=" << t_les
              << "s\n";
}

// (slot-25 GOLD-PORT; RETIRED by the HZ-GOLD landing 2026-11-07) The scratch interlock GOLDEN_GEVT23=1 and its GOLDEN_OUT
// refusal (GEvt23ScratchRequested / GEvt23EmitRefusal) existed only while the gates were false; they are removed, and the
// variable is no longer read.  History: docs/implementation/slot25/GOLDPORT_report.md, golden_ledger.md.

// ===========================================================================
// (slot-25 HZ-GOLD; docs/implementation/slot25/golden_ledger.md, expect set docs/implementation/slot25/hzgold/expect_hz15.tsv;
// slot24/heckez/04-record-census.md Sec. 1.4, 06-work-packages.md HZ-GOLD) THE HZ POINTS -- the integral Hecke structure
// record integral_hecke_structure/v1 (Integral/HeckeZRecord.h; [C] sec:recordschema item 8) at the 15 level-1 cells of the
// first golden set: (1, n), n = 2, 4, ..., 30, algebra "full" (= the anemic algebra at N = 1).  Looped from ComputeAll()
// ONLY (after the G-EVT23 points), never from RunCell32 / RunExtCell32 / the console scripts -- so GoldenSubset32
// (EngineDeterminismValidation), the ext records (HeckeExtDeterminismValidation) and the console C4 (a) scripts cannot
// reach them.
//
// Point string `HZ N=<N> n=<n> alg=<alg>` (new; used by no pre-existing line).  ONE object per cell, TIER A:
//   HZ_N<N>_n<n>_<alg>      raw: rows 1, cols 1, hash = the record's record_fnv1a64 (the one FNV-1a-64 of Fnv.h over the
//                           record's invariant body, 04 Sec. 1.4: a function of (cell, algebra, settings), independent of the
//                           lattice basis and of algorithm internals; the certificate layer G9 is outside it).  rows / cols
//                           carry no information (fixed 1 x 1: the object is one 64-bit value); the hash is the record's own
//                           hex value parsed, so it can be compared with HZREC_report.md Sec. 10.2 by eye.
// The record is built with the default RecordSettings and DefaultRecordSources() under the ALL-DEFAULTS MathSettings with
// coefficient_ring = Z (ScopedMathSettings(defaults + coefficient_ring.k = 0, source spec)), exactly the settings under which
// the campaign arm hecke_structure and the console verb build the default record.  (HZ-GOLD code audit B1) NOT
// ScopedCoefficientRing(0), which keeps the caller's other keys: the record's settings text -- and so its record_fnv1a64 --
// carries the GLOBAL MathSettings canonical text, so under HeckeLocalitySlowValidation's strict-locality sweep
// (GoldenLocalitySweep -> ComputeAll) the record would be a different, non-default record.  Only defaults are golden
// (04 Sec. 1.4) and the sweep never compares HZ objects, so the HZ points pin the defaults whatever the caller installed;
// value-neutral at the defaults.  A line exists only for a record whose overall status is final, with heckez.G8 = off
// (golden_eligible(): it checks exactly these two, NOT the settings -- audit N1) AND whose settings fingerprint is the
// default kHZDefaultFingerprint (04 Sec. 1.4: only defaults are golden; this is the settings check) -- ENGINE_CHECK, so a
// record that turns pending or a changed default fails loudly instead of emitting a different line.  No expected value is asserted
// here (the 15 values are pinned in HeckeHidaEisLambdaValidation (C) and, after the landing, by this reference).
//
// (HZ-GOLD referee F3) THE OBJECT IS A HASH OF THE WHOLE RECORD, settings and caps text included: a change to the record
// schema, to any word of MathSettings::canonical_text() / RecordSettings::caps_text() (a new default key, a cap, a reworded
// rule), to a record source's output or a new hashed field moves all 15 hashes -- and, where it moves the settings text, the
// fingerprint ENGINE_CHECK above THROWS in every golden pass at once (GoldenRegressionValidation, HeckeLocality(Slow)Validation,
// ConsoleDeterminismValidation arm (c), GoldenEmit).  That is an OWNER-GATED 15-line "changed" re-baseline: kHZDefaultFingerprint,
// the HeckeHidaEisLambdaValidation pins and the HZ-DS dataset ids move in the same commit.  Diagnose by rebuilding the record
// with the console verb hecke_structure(1, n) and diffing its JSON against the record's member of the HZ-DS ed.1
// records_bundle.tar.gz (docs/integral_census/results/hecke_ed1/); the 1 x 1 raw object itself says nothing about which field
// moved (golden_explain.py and GoldenDump do not see it).
//
// THE GATE: kHZInReference -- TRUE since the HZ-GOLD landing (2026-11-07), together with kGEvt23InReference: the two groups
// are one event (one provenance paragraph, k = 112), so the static_assert below ties them (HZ-GOLD code audit S1).  (Before the
// landing both were false under the scratch interlock GOLDEN_GEVT23=1, now retired.)  The gate selects WHICH records are
// computed, never a value.
//
// LANDING: 2026-11-07, the add-only event of slot 25, k = 112 (docs/implementation/slot25/golden_ledger.md Sec. 7)
// ===========================================================================
constexpr bool kHZInReference = true;   // HZ-GOLD landing, 2026-11-07
static_assert(kGEvt23InReference == kHZInReference, "HZ-GOLD: the two groups land in one commit (one provenance paragraph, k = 112)");
constexpr const char* kHZDefaultFingerprint = "0x8ffc29730c4f9672";   // HZREC_report.md Sec. 10.2 (the frozen default)
const std::vector<uint32_t> kGridHZ = {2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30};   // N = 1, alg = full

void RunHZPoint(Collector& C, uint32_t N, uint32_t n, const std::string& alg) {
    namespace hzr = heckez::record;
    const std::string pt = "HZ N=" + std::to_string(N) + " n=" + std::to_string(n) + " alg=" + alg;
    Timer T;
    // (code audit B1) the all-defaults settings with coefficient_ring = Z, whatever the caller (e.g. the locality sweep) installed
    hecke::MathSettings ms = hecke::MathSettings::defaults();
    ms.coefficient_ring.k = 0;   // Z: the record is an object over Z ([C] sec:recordschema item 8 (a))
    const hecke::ScopedMathSettings ring(ms, hecke::SettingSource::Spec);
    const hzr::HeckeStructureRecord R = hzr::BuildHeckeStructureRecord(N, n, alg, hzr::RecordSettings{}, hzr::DefaultRecordSources());
    ENGINE_CHECK_MSG(R.N == N && R.n == n && R.algebra == alg, (std::string("golden HZ: the record is not the cell's at ") + pt).c_str());
    ENGINE_CHECK_MSG(R.status == "final" && R.golden_eligible(),
                     (std::string("golden HZ: the record is not golden-eligible (status ") + R.status + ") at " + pt).c_str());
    ENGINE_CHECK_MSG(R.settings_fingerprint == kHZDefaultFingerprint,
                     (std::string("golden HZ: settings fingerprint ") + R.settings_fingerprint + " is not the default " +
                      kHZDefaultFingerprint + " at " + pt)
                         .c_str());
    ENGINE_CHECK_MSG(R.record_fnv1a64.size() == 18 && R.record_fnv1a64.compare(0, 2, "0x") == 0,
                     (std::string("golden HZ: record_fnv1a64 '") + R.record_fnv1a64 + "' is not 0x + 16 hex digits at " + pt).c_str());
    bool hex = true;   // (code audit N3) 16 hex digits, so the parse below cannot throw
    for (size_t i = 2; i < R.record_fnv1a64.size(); ++i) hex = hex && std::isxdigit(static_cast<unsigned char>(R.record_fnv1a64[i])) != 0;
    ENGINE_CHECK_MSG(hex, (std::string("golden HZ: record_fnv1a64 '") + R.record_fnv1a64 + "' has a non-hex digit at " + pt).c_str());
    const uint64_t h = std::stoull(R.record_fnv1a64.substr(2), nullptr, 16);
    ENGINE_CHECK_MSG(HexFnv(h) == R.record_fnv1a64,
                     (std::string("golden HZ: record_fnv1a64 '") + R.record_fnv1a64 + "' does not round-trip at " + pt).c_str());
    C.raw(pt, "HZ_N" + std::to_string(N) + "_n" + std::to_string(n) + "_" + alg, 'A', 1, 1, h);
    std::cout << "  [" << pt << "] record_fnv1a64=" << R.record_fnv1a64 << " status=" << R.status << " fp=" << R.settings_fingerprint
              << " " << std::fixed << std::setprecision(2) << T.secs() << " s\n";
}

void RunHZPoints(Collector& C) {
    Timer T;
    std::cout << "--- HZ points (integral_hecke_structure/v1, coefficient_ring = Z; the first golden set) ---\n";
    for (uint32_t n : kGridHZ) RunHZPoint(C, 1, n, "full");
    std::cout << "  HZ wall: " << std::fixed << std::setprecision(2) << T.secs() << " s\n";
}

// Computes every golden record in the fixed emission order.  with_gevt23: the G-EVT23 families (see kGEvt23InReference);
// with_hz: the HZ points (see kHZInReference).
std::vector<Record> ComputeAll(bool with_gevt23, bool with_hz) {
    Collector C;
    Timer T;
    const SimplicialComplex K = CreateSL2ZFundamentalDomainTriangulation();
    const hecke::SL2ZTessellationTables model(K);

    std::cout << "--- 32-bit grid ---\n";
    for (uint32_t n : kGridNSym)
        for (uint32_t ell : kGridEll) RunCell32(C, K, model, n, ell);
    for (const auto& ne : kExtraCells) RunCell32(C, K, model, ne.first, ne.second);

    std::cout << "--- 64-bit points ---\n";
    for (uint64_t ell : kGridEll64)
        for (uint32_t n : kGridNSym64) RunCell64(C, K, model, ell, n);

    std::cout << "--- S6 induced maps ---\n";
    RunS6Presets(C);
    for (const auto& ne : kGridS6Restriction) RunS6Restriction(C, K, ne.first, ne.second);

    // (slot-8 WP-4) The E8 points: from here only, after the S6 loop.
    std::cout << "--- E8 points (coefficient_ring = Z/ell^k) ---\n";
    for (const E8Point& e8 : kGridE8) RunE8Point(C, K, model, e8);

    // (slot-13 WP-5) The Hida points: from here only, after the E8 loop.
    std::cout << "--- HIDA points (the first Hida emission) ---\n";
    for (const HidaPoint& hp : kGridHida) RunHidaPoint(C, hp);

    // (slot-17 WP-4) The level-N, LOEWY_RING and nebentypus points: from here only, after the HIDA loop.
    std::cout << "--- LEVELN points (level N, kappa_N, alpha_d) ---\n";
    for (const LevelNPoint& lp : kGridLevelN) RunLevelNPoint(C, K, lp);
    std::cout << "--- LOEWY_RING points (certified, cor:stablecert) ---\n";
    for (const LoewyPoint& lp : kGridLoewy) RunLoewyPoint(C, lp);
    std::cout << "--- NEB points (nebentypus, Conrey labels) ---\n";
    for (const NebPoint& np : kGridNeb) RunNebPoint(C, np);
    // (slot-18 WP-4) The p != 2 tau / tau_bd tables: from here only, after the NEB loop.
    std::cout << "--- ext tau points (tau / tau_bd at p != 2, rmk:iotatausym) ---\n";
    for (uint32_t n : kGridNSym)
        for (uint32_t ell : kGridEll)
            if (InExtSubgrid(n, ell)) RunExtTauCell(C, K, model, n, ell);
    for (const auto& ne : kExtraCells)
        if (InExtSubgrid(ne.first, ne.second)) RunExtTauCell(C, K, model, ne.first, ne.second);
    // (slot-22 G-EVT) The L-value and Hida pairing points: from here only, after the ext tau loop.
    std::cout << "--- LVALUE points (v(#Q_m), the cross-kind depth, c_f^eps) ---\n";
    for (const LValuePoint& lp : kGridLValue) RunLValuePoint(C, lp);
    std::cout << "--- HIDAPAIR points (the 27 Hida pairing valuation tables) ---\n";
    for (const auto& ke : kGridHidaPair) RunHidaPairPoint(C, ke.first, ke.second);
    // (slot-24 D6, the ONE add-only event of the integral descent; docs/implementation/slot24/d6golden/ledger.md) The DESCENT
    // points: from here only, after the HIDAPAIR loop.
    std::cout << "--- DESCENT points (integral_descent_record/v1, coefficient_ring = Z) ---\n";
    RunDescentPoints(C);
    // (slot-23 G-EVT23, ported in slot 25 GOLD-PORT) The TAMEFAM, PRANK, TOWER, SHAPIRO and LESLEVEL points: from here only,
    // after the DESCENT loop.
    if (with_gevt23) {
        RunGEvt23Points(C);
    } else {
        std::cout << "--- G-EVT23 points (TAMEFAM / PRANK / TOWER / SHAPIRO / LESLEVEL): NOT computed (kGEvt23InReference = false) ---\n";
    }
    // (slot-25 HZ-GOLD) The HZ points: from here only, after the G-EVT23 points.
    if (with_hz) {
        RunHZPoints(C);
    } else {
        std::cout << "--- HZ points (integral_hecke_structure/v1): NOT computed (kHZInReference = false) ---\n";
    }

    std::cout << "Golden: computed " << C.records().size() << " objects in " << std::fixed
              << std::setprecision(1) << T.secs() << " s\n";
    return C.records();
}

// ===========================================================================
// Reference I/O.
// ===========================================================================

std::string DefaultReferencePath() {
    std::string f = __FILE__;
    const size_t slash = f.find_last_of("/\\");
    const std::string dir = (slash == std::string::npos) ? std::string(".") : f.substr(0, slash);
    return dir + "/GoldenReference.inc";
}

std::string Hex64(uint64_t v) {
    std::ostringstream oss;
    oss << "0x" << std::hex << std::setw(16) << std::setfill('0') << v << "ull";
    return oss.str();
}

void WriteReference(const std::vector<Record>& recs, const std::string& path) {
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    if (!out) throw std::runtime_error("GoldenEmit: cannot open " + path + " for writing");
    out << "// GoldenReference.inc -- GENERATED by `./cohomology_app GoldenEmit`. DO NOT EDIT BY HAND.\n";
    out << "// " << kProvenanceNote << "\n";
    out << "// Emitting build: compiler " << __VERSION__
#ifdef NDEBUG
        << ", NDEBUG"
#else
        << ", asserts on"
#endif
#ifdef __OPTIMIZE__
        << ", optimised"
#else
        << ", -O0"
#endif
#ifdef HECKE_PURE_INTEGER
        << ", HECKE_PURE_INTEGER=ON"
#else
        << ", HECKE_PURE_INTEGER=OFF"
#endif
        << ".\n";
    out << "// Format: {point, object, tier, rows|value, cols, FNV-1a-64}; tier 'A' = acceptance\n";
    out << "// (basis-independent, must never change), 'B' = change detector (basis-dependent;\n";
    out << "// fails only under GOLDEN_STRICT=1). See GoldenHarness.cpp for the definitions.\n";
    out << "// Entries: " << recs.size() << "\n";
    for (const Record& r : recs) {
        out << "{\"" << r.point << "\", \"" << r.object << "\", '" << r.tier << "', " << r.rows << ", "
            << r.cols << ", " << Hex64(r.hash) << "},\n";
    }
    if (!out) throw std::runtime_error("GoldenEmit: write failure on " + path);
}

} // namespace

// (slot-24 D6) The staged DESCENT points, in emission order -- GoldenHarness.h.
std::vector<GoldenComputedRecord> GoldenDescentRecords() {
    Collector C;
    RunDescentPoints(C);
    std::vector<GoldenComputedRecord> out;
    for (const Record& r : C.records()) out.push_back(GoldenComputedRecord{r.point, r.object, r.tier, r.rows, r.cols, r.hash});
    return out;
}

// ===========================================================================
// (hecke-extensions slot 6, WP-6 / E2 close-out) Exposed harness recipes --
// GoldenHarness.h.  HeckeExtCampaignValidation's golden arm needs to hash the
// campaign's own in-memory `hecke_ext::HeckeExtRecord` with EXACTLY the
// recipes GoldenRegressionValidation's ext sweep uses (above), and
// HeckeExtDeterminismValidation needs the same `commit` redaction the golden
// sweep already applies ([X] Sec. 1.6; slot-5 WP-3 finding 3). These are thin
// forwarders onto the anonymous-namespace definitions above: there remains
// exactly ONE definition of each recipe, never a second, hand-copied one.
// ===========================================================================
std::string GoldenNormalizeJsonWhitespace(const std::string& s) {
    return NormalizeJsonWhitespace(s);
}
std::string GoldenRedactJsonKeys(const std::string& normalized,
                                 const std::vector<std::string>& keys) {
    return RedactJsonKeys(normalized, keys);
}
std::string GoldenReduceSettingsToFingerprint(const std::string& normalized) {
    return ReduceSettingsToFingerprint(normalized);
}
uint64_t GoldenHashExtIdeals(const hecke_ext::HeckeExtRecord& r) { return HashExtIdeals(r); }
uint64_t GoldenHashExtCertify(const hecke_ext::HeckeExtRecord& r) { return HashExtCertify(r); }
uint64_t GoldenHashSturmBound(hecke_ext::SturmBound& B) { return HashSturmBound(B); }

// ===========================================================================
// (console phase C1a) The ONE read-only lookup over GoldenReference.inc --
// GoldenHarness.h, inventory gap G13, [X] Sec. 1.1.
//
// It is the pre-C1a `FindReference` body, moved verbatim: a std::map built once
// (no hash container, audit Sec. 6.5(5)) keyed by "<point>|<object>", over the
// compiled-in array.  Every caller -- GoldenLocalitySweep's hash re-check and
// the console's `golden(point, object)` verb -- goes through this function, so
// a console pin and the regression validation read the identical bytes.
// ===========================================================================

const GoldenEntry* GoldenLookup(const std::string& point, const std::string& object) {
    static std::map<std::string, const GoldenEntry*> idx = [] {
        std::map<std::string, const GoldenEntry*> m;
        for (const GoldenEntry* e = kGoldenReference; e->point != nullptr; ++e)
            m[std::string(e->point) + "|" + e->object] = e;
        return m;
    }();
    auto it = idx.find(point + "|" + object);
    return it == idx.end() ? nullptr : it->second;
}

size_t GoldenReferenceSize() {
    size_t n = 0;
    for (const GoldenEntry* e = kGoldenReference; e->point != nullptr; ++e) ++n;
    return n;
}

// ===========================================================================
// Registered entry points.
// ===========================================================================

uint32_t GoldenSubset32(const std::vector<std::pair<uint32_t, uint32_t>>& cells, uint64_t& digest,
                        size_t& records, bool quiet) {
    std::map<std::string, const GoldenEntry*> ref;
    for (const GoldenEntry* e = kGoldenReference; e->point != nullptr; ++e)
        ref[std::string(e->point) + "|" + e->object] = e;
    Collector C;
    {
        // Silence the per-cell progress lines when asked (they carry timings).
        std::streambuf* saved = quiet ? std::cout.rdbuf() : nullptr;
        std::ostringstream sink;
        if (quiet) std::cout.rdbuf(sink.rdbuf());
        try {
            const SimplicialComplex K = CreateSL2ZFundamentalDomainTriangulation();
            const hecke::SL2ZTessellationTables model(K);
            for (const auto& ne : cells) RunCell32(C, K, model, ne.first, ne.second);
        } catch (...) {
            if (quiet) std::cout.rdbuf(saved);
            throw;
        }
        if (quiet) std::cout.rdbuf(saved);
    }
    uint32_t fails = 0;
    uint64_t h = kFnvOffset;
    for (const Record& r : C.records()) {
        h = FnvBytes(h, r.point); h = FnvBytes(h, r.object);
        h = FnvU64(h, r.rows); h = FnvU64(h, r.cols); h = FnvU64(h, r.hash);
        auto it = ref.find(r.point + "|" + r.object);
        if (it == ref.end()) { ++fails; std::cout << "  GoldenSubset32: MISSING from reference: " << r.point << " / " << r.object << "\n"; continue; }
        const GoldenEntry& e = *it->second;
        if (!(e.rows == r.rows && e.cols == r.cols && e.hash == r.hash && e.tier == r.tier)) {
            ++fails;
            std::cout << "  GoldenSubset32: MISMATCH [tier " << e.tier << "] " << r.point << " / " << r.object
                      << ": expected " << Hex64(e.hash) << " got " << Hex64(r.hash) << "\n";
        }
    }
    digest = h;
    records = C.records().size();
    return fails;
}

// ===========================================================================
// (hecke-extensions phase E0a) GoldenDump + GoldenLocalitySweep.
// ===========================================================================

uint32_t GoldenDump() {
    try {
        std::cout << "================= GoldenDump =================\n";
        // Safety interlock, the GoldenEmit pattern: GoldenDump is a registered
        // name and would otherwise run (and write hundreds of megabytes) under
        // `--all`. It only writes when the caller opts in explicitly.
        const char* dir_env = std::getenv("GOLDEN_DUMP");
        if (dir_env == nullptr || std::string(dir_env).empty()) {
            std::cout << "GoldenDump: NOT dumping (safety interlock). To write the object dump run\n"
                         "    GOLDEN_DUMP=<dir> ./hacs GoldenDump\n"
                         "  It writes <dir>/index.tsv, <dir>/texts.tsv and <dir>/decompositions.txt,\n"
                         "  the 'before'/'after' inputs of docs/matrix-engine/tools/golden_explain.py.\n"
                         "  Returning 0 so that `--all` stays a pure read-only run.\n";
            return 0;
        }
        const std::string dir = dir_env;
        std::error_code ec;
        std::filesystem::create_directories(dir, ec);
        if (ec) throw std::runtime_error("GoldenDump: cannot create " + dir + ": " + ec.message());

        DumpSink sink;
        sink.index.open(dir + "/index.tsv", std::ios::binary | std::ios::trunc);
        sink.texts.open(dir + "/texts.tsv", std::ios::binary | std::ios::trunc);
        sink.decomps.open(dir + "/decompositions.txt", std::ios::binary | std::ios::trunc);
        if (!sink.index || !sink.texts || !sink.decomps) {
            throw std::runtime_error("GoldenDump: cannot open the output files under " + dir);
        }
        sink.index << "# golden-dump v1\tkind\tpoint\tobject\n";
        sink.texts << "# golden-dump v1: point\tobject\tnormalised-json\n";
        sink.decomps << "# golden-dump v1: block lists (the inputs of _invariants / _certify /"
                        " _witness / _U)\n";

        g_dump = &sink;
        std::vector<Record> recs;
        try {
            recs = ComputeAll(kGEvt23InReference, kHZInReference);
        } catch (...) {
            g_dump = nullptr;
            throw;
        }
        g_dump = nullptr;
        sink.index.flush();
        sink.texts.flush();
        sink.decomps.flush();
        if (!sink.index || !sink.texts || !sink.decomps) {
            throw std::runtime_error("GoldenDump: write failure under " + dir);
        }
        std::cout << "GoldenDump: wrote " << sink.n_texts << " text objects and " << sink.n_decomps
                  << " decompositions (" << recs.size() << " golden records computed) to " << dir
                  << "\n";
        return 0;
    } catch (const std::exception& e) {
        std::cout << "GoldenDump: TOP-LEVEL EXCEPTION: " << e.what() << std::endl;
        return 1;
    }
}

GoldenLocalitySweepResult GoldenLocalitySweep(bool quiet) {
    GoldenLocalitySweepResult R;
    g_sweep = &R;
    std::streambuf* saved = nullptr;
    std::ostringstream sink;
    if (quiet) { saved = std::cout.rdbuf(); std::cout.rdbuf(sink.rdbuf()); }
    try {
        ComputeAll(kGEvt23InReference, kHZInReference);
    } catch (...) {
        if (quiet) std::cout.rdbuf(saved);
        g_sweep = nullptr;
        throw;
    }
    if (quiet) std::cout.rdbuf(saved);
    g_sweep = nullptr;
    return R;
}

uint32_t GoldenEmit() {
    try {
        const char* env = std::getenv("GOLDEN_OUT");
        const std::string path = env ? std::string(env) : DefaultReferencePath();
        std::cout << "================= GoldenEmit =================\n";
        std::cout << "*** WARNING: this OVERWRITES the golden acceptance reference\n"
                  << "***   " << path << "\n"
                  << "*** Re-baselining tier-A objects is an owner decision; every changed line\n"
                  << "*** must be explained object-by-object (docs/matrix-engine/00 Sec. 4.1).\n";
        // Safety interlock: GoldenEmit is a registered name and would otherwise run
        // (and rewrite the checked-in reference) under `--all`. It only emits when
        // the caller opts in explicitly:  GOLDEN_EMIT=1 ./cohomology_app GoldenEmit
        const char* confirm = std::getenv("GOLDEN_EMIT");
        if (!(confirm && std::string(confirm) == "1")) {
            std::cout << "GoldenEmit: NOT emitting (safety interlock). To regenerate the reference run\n"
                         "    GOLDEN_EMIT=1 ./hacs GoldenEmit\n"
                         "  (optionally GOLDEN_OUT=<path> to write elsewhere). Returning 0 so that\n"
                         "  `--all` stays a pure read-only run.\n";
            return 0;
        }
        const std::vector<Record> recs = ComputeAll(kGEvt23InReference, kHZInReference);
        WriteReference(recs, path);
        uint32_t nA = 0, nB = 0;
        for (const Record& r : recs) (r.tier == 'A' ? nA : nB)++;
        std::cout << "GoldenEmit: wrote " << recs.size() << " entries (" << nA << " tier A, " << nB
                  << " tier B) to " << path << "\n"
                  << "GoldenEmit: rebuild (make) so GoldenRegressionValidation picks up the new reference.\n";
        return 0;
    } catch (const std::exception& e) {
        std::cout << "GoldenEmit: TOP-LEVEL EXCEPTION: " << e.what() << std::endl;
        return 1;
    }
}

uint32_t GoldenRegressionValidation() {
    try {
        std::cout << "================= GoldenRegressionValidation =================\n";
        const char* strict_env = std::getenv("GOLDEN_STRICT");
        const bool strict = strict_env && std::string(strict_env) == "1";
        std::cout << "Tier B policy: " << (strict ? "STRICT (GOLDEN_STRICT=1): tier-B mismatches FAIL"
                                                  : "report only (set GOLDEN_STRICT=1 to fail on tier B)")
                  << "\n";

        // Load the compiled-in reference.
        std::map<std::string, const GoldenEntry*> ref;
        size_t n_ref = 0;
        for (const GoldenEntry* e = kGoldenReference; e->point != nullptr; ++e) {
            ref[std::string(e->point) + "|" + e->object] = e;
            ++n_ref;
        }
        if (n_ref == 0) {
            std::cout << "  *** FAIL: GoldenReference.inc is EMPTY. Run `./hacs GoldenEmit`, "
                         "rebuild, and re-run.\n";
            return 1;
        }
        std::cout << "Reference entries: " << n_ref << "\n";

        const std::vector<Record> recs = ComputeAll(kGEvt23InReference, kHZInReference);

        uint32_t failA = 0, failB = 0, checkedA = 0, checkedB = 0;
        std::map<std::string, bool> seen;
        for (const Record& r : recs) {
            const std::string key = r.point + "|" + r.object;
            seen[key] = true;
            auto it = ref.find(key);
            if (it == ref.end()) {
                ++failA;
                std::cout << "  *** FAIL [tier A] point '" << r.point << "' object '" << r.object
                          << "': computed but MISSING from the reference (grid/object set changed?)\n";
                continue;
            }
            const GoldenEntry& e = *it->second;
            (e.tier == 'A' ? checkedA : checkedB)++;
            const bool ok = (e.rows == r.rows && e.cols == r.cols && e.hash == r.hash && e.tier == r.tier);
            if (!ok) {
                const bool counts = (e.tier == 'A') || strict;
                if (e.tier == 'A') ++failA; else ++failB;
                std::cout << "  *** " << (counts ? "FAIL" : "MISMATCH") << " [tier " << e.tier << "] point '"
                          << r.point << "' object '" << r.object << "': expected dims (" << e.rows << ","
                          << e.cols << ") hash " << Hex64(e.hash) << ", got dims (" << r.rows << ","
                          << r.cols << ") hash " << Hex64(r.hash) << (e.tier != r.tier ? " (tier changed)" : "")
                          << "\n";
            }
        }
        for (const auto& kv : ref) {
            if (!seen.count(kv.first)) {
                ++failA;
                std::cout << "  *** FAIL [tier A] reference entry point '" << kv.second->point << "' object '"
                          << kv.second->object << "' was NOT computed by this binary\n";
            }
        }

        std::cout << "GoldenRegressionValidation: tier A checked " << checkedA << ", failing " << failA
                  << "; tier B checked " << checkedB << ", mismatching " << failB
                  << (strict ? " (counted as failures)" : " (reported only)") << "\n";
        if (failB && !strict)
            std::cout << "  NOTE: tier-B changes mean a basis convention changed (D5b). Re-baseline only with\n"
                         "  an owner-approved, object-by-object note; run with GOLDEN_STRICT=1 to enforce.\n";
        return failA + (strict ? failB : 0u);
    } catch (const std::exception& e) {
        std::cout << "GoldenRegressionValidation: TOP-LEVEL EXCEPTION: " << e.what() << std::endl;
        return 1;
    }
}
