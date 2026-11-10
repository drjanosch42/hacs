// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

#pragma once
#ifndef CAMPAIGN_ORBIFOLD_H
#define CAMPAIGN_ORBIFOLD_H

// ===========================================================================
// (hecke-extensions slot 22, L23-1) THE ORBIFOLD SECTION OF THE CAMPAIGN:
// `orbifold = c_i` -- one `orbifold_cell/v2` record per (n_sym, ell) cell (v1 before S24 L23-H1),
// ell in {2, 3}, of ORDINARY GROUP COHOMOLOGY H^q(SL_2(Z); Sym^n F_ell^2),
// q <= 3, and of the ell-adic lattice of H^1(SL_2(Z); Sym^n Z^2).  Never
// Bredon: at ell in {2, 3} the shipped field pipeline and the ext record are
// Bredon objects, and nothing here touches them.
//
// Binding sources: [C] = docs/latex/gl2_contract.tex 1.22.0-pre.11 (sec:oddnltwo:
// lem:centralminusone, lem:bocksteinallell, prop:oddnltwo, lem:oddnsplit,
// lem:tellrank, prop:ltwothreescalar, prop:oddnhecke, lem:cprule;
// prop:isomodel, prop:strictlift, lem:bulkperiodic, cl:amalgam,
// prop:certifiede).  Plan: docs/implementation/slot22/census_l23_plan.md v0.3
// ([P3]) Sec. 2 (schema), Sec. 6 (L23-1), as amended by L23_master_plan.md
// v0.2 ([MP]: BIN-3 is the reader; the odd-n gates of Sec. 1).  Report:
// docs/implementation/slot22/L23_1_report.md.  User documentation:
// docs/matrix-engine/campaign.md Sec. 16.
//
// WHAT A CELL COMPUTES (all single-process, deterministic):
//   (F)  the F_ell part on the isotropy-resolution model C_I ([C] prop:isomodel(3):
//        H^q(C_I (x) R) = H^q(Gamma; V_R) for every ring R), q_max = 3:
//        h^0 .. h^3; T_p = Theta^_{diag(1,p)} ([C] prop:strictlift(3)) on every
//        H^q as a matrix; the gate T_p - (1 + p) = 0 for p != ell
//        ([C] prop:ltwothreescalar); the Jordan type of T_ell per degree and on
//        the (U) layers H^q(V)/ell (sub) and T^{q+1}[ell] (quotient), q = 1, 2;
//        e u - : H^0 -> H^2 and H^1 -> H^3 ([C] lem:bulkperiodic) and its
//        Hecke equivariance.
//   (Z)  the same model over Z/ell^K: the torsion of H^q(V) is the torsion of
//        coker(delta^{q-1}) (Z^q is saturated in the free C^q), read as the
//        Smith invariants of delta^{q-1} -- K = 31 / 19 at q = 1 (an independent
//        cross-check of the reader's T^1), K = E = 4 / 2 at q = 2, 3 (one above
//        the exponent bounds 3 / 1 of [C] cl:amalgam(4); ladic_type_2 / _3);
//        and H^q(V)/ell inside H^q(M) for the (U) layers, q = 1, 2, as the
//        reduction of the cocycles mod ell^E (E - 1 >= exp T^{q+1}; the UCT
//        reading of IntegralHecke.h).  The counts t2, t3 are gated against the
//        UCT dimension identities.
//   (L)  the degree-1 amalgam reader l23::read_lattice (BIN-3): rank and
//        torsion of H^1(Gamma; V)_ell, the elementary divisors of
//        T_p - (1 + p), T_p - (1 + p^{n+1}), T_ell on Lambda^1 (x) Z_ell, and at
//        ell = 3 the same on Lambda^+ / Lambda^- ([C] def:iota); the
//        precision promise m >= m_target, m_target = max(floor((n+1)/(ell+1)) + 10,
//        40) by default ([P3] Sec. 3.3, [MP] Sec. 2.2); c_p = min ed_tp with the
//        [C] lem:cprule identity as a gate.
//   (A)  the independent amalgam oracle (L23-1 audit R3; spec docs/implementation/slot22/fl231/compare_h1mod.py):
//        l23::h1_mod(ell, n, k) -- H^1(Gamma; V/ell^k) on the amalgam Z/4 *_{Z/2} Z/6, sharing no code with C_I --
//        at k = 1 with T_ell, T_ell - 1, and L_k = log_ell |H^1(V/ell^k)| for k <= 4 / 2; cross-model G gates on
//        h^0, h^1, rank T_ell and rank (T_ell - 1) on H^1(M), and L_k = k rank + sum min(k, a_i) + sum min(k, b_j)
//        (a = T^1, b = T^2), which pins the T^2 multiset given T^1.  ell = 2: n >= 1; ell = 3: even n >= 2.
//   (S)  (L23-3) the level-1 q-expansion sum rules (l23::sum_rule_oracle; independent of the amalgam, the reader and
//        C_I): sum ed_tell = 2 v det(T_ell | S_k), sum ed_tp_p = 2 v det(T_p - (1+p) | S_k) + v(p^n - 1), sum of the finite
//        ed_eis_p = 2 v det(T_p - (1+p^{n+1}) | S_k), and at ell = 3 the same on Lambda^+- (v det, + v(lambda_Eis) on
//        Lambda^+) -- G gates on every certified lattice cell (census_l23_plan.md Sec. 4.3; Eichler--Shimura).
//   (B/!/L) (L23-3) l23::h1_layers on the amalgam over F_ell: dim H^1_Br and the T_ell (B) triple (sub / quot / whole),
//        dim H^1(V)/ell and dim T^1/ell (cross-model with C_I and the reader), dim H^1_! (Gamma_inf = +-<T>) and its
//        intersections with H^1_Br, H^1(V)/ell, T^1/ell; at ell = 3 the inflated line L, L cap H^1_Br = 0, L in T^1/3.
//        Gates: [R] Thm B(5) / C(1) (codim H^1_Br), Thm E (Hecke-stable; = ker e u at ell = 3), Thm C(3) / C(4) (L).
//   Gates are of two kinds ([P3] Sec. 2): `G` (a theorem: a mismatch FAILS the
//   cell -- a bug or a refutation; the campaign STOPS there, [P3] R7) and
//   `G-conj` (a conjecture: a mismatch is a FINDING, recorded; the cell does
//   not fail).  The divisor convention ([P3] Sec. 2.3): a unit prints `0`, the
//   (known) kernel `inf`, a divisor not certified below the cap `>=<cap>`.
//
// DISCIPLINE.  refused: every cell this build does not execute is refused at
// spec validation (ValidateOrbifoldSpec), by name, before any cell runs, so a
// run that starts reads refused=0.  fallback: a cell whose reader could not
// certify a divisor that should be finite (anything `>=cap` beyond the one
// Eisenstein kernel of ed_eis), or that reached orbifold_k_cap unresolved, is
// recorded with status `fallback` and counted; the final line reads
// `EFFECTIVE campaign arm=orbifold cells=<k> refused=<r> fallback:<f> ...`.
//
// IDENTITY.  cell id `orb_ell<L>_n<N>`, plus `_s<16 hex>` when
// OrbifoldFingerprint(spec) != 0: the FNV-1a-64 of the non-default
// mathematical settings (hecke::MathSettings, recorded and not consulted by
// the cell) and of the non-default orbifold keys (orbifold_primes,
// orbifold_m_target, orbifold_k_cap), which may change a recorded value.  The
// record's `record_fnv1a64` is the FNV-1a-64 of its canonical body, which
// carries the `settings` block -- so a non-default setting changes the hash
// input as well as the id.  Timings and RSS are outside the body.
//   (L23-1 audit R4) So is the reader's WORKING-PRECISION PATH -- rung, K, the losses e_A / e_1, reruns and the
// certified precision m reached -- in the non-hashed `reader_path` block.  The mathematical content does not
// depend on that path: every divisor below the cap is an invariant of the Z_ell-module (the elementary divisors
// of an operator on Lambda^1 (x) Z_ell, the exponents of H^1(V)_ell), and `certified` asserts that each was read
// at a certified precision m >= m_target = cap; a divisor not certified below the cap prints `>=<cap>` whatever K
// was.  The hashed `reader` block keeps what the certification is relative to: ran, ok, m_target, cap, certified.
// So a different k_min, instantiated width set (another build) or ladder change leaves record_fnv1a64 unchanged
// unless a value moves.  Provenance (commit, binary sha256; audit R6) is outside the body too.
//
// RESUME ([P3] R7, L23-1 audit R1).  `resume = true` skips a complete record only if its status is `ok` or
// `fallback`; a complete `failed` record (or any other status) stops the campaign there with `stopped=1`, its
// `failed` carried through to summary.tsv / index.json, exit 2 -- the owner removes or renames the file first.
// ===========================================================================

#include "CampaignSpec.h"
#include "../../ModArithmetic/DenseBuf.h"

#include <cstdint>
#include <map>
#include <string>
#include <vector>

#include "../../ModArithmetic/EnvLever.h"

namespace integral_model { class IsotropyModel; class StrictLift; }

namespace campaign {

// The default precision promise ([P3] Sec. 3.3): max(floor((n+1)/(ell+1)) + 10, 40).
uint32_t OrbifoldAutoMTarget(uint32_t ell, uint32_t n);
// The m_target a cell of this spec runs under.
uint32_t OrbifoldMTarget(const CampaignSpec& spec, uint32_t ell, uint32_t n);
// 0 at the defaults (no non-default MathSettings entry, primes 2,3,5,7, m_target auto, k_cap 0).
uint64_t OrbifoldFingerprint(const CampaignSpec& spec);
// The cell id: "orb_ell2_n10" or "orb_ell2_n10_s0123456789abcdef".
std::string OrbifoldCellId(const CampaignSpec& spec, uint32_t ell, uint32_t n);

struct OrbifoldCellPlan {
    uint32_t ell = 0, n = 0;
    std::string id;
};
// Union of the sections' (n_sym x ell) products, de-duplicated, ordered by (ell, n).
std::vector<OrbifoldCellPlan> PlanOrbifoldCells(const CampaignSpec& spec);

// A Jordan type over F_ell restricted to the eigenvalues 0 and 1 (block sizes, descending); `other` = the
// dimension of the remaining generalised eigenspaces (eigenvalues outside {0, 1} or outside F_ell).
struct JordanType {
    uint32_t dim = 0, rank = 0;
    std::vector<uint32_t> eig0, eig1;
    uint32_t other = 0;
    std::string text() const;   // "0:[2,1] 1:[1]" ("-" for the zero space); " other:<d>" when other > 0
    bool rank_one_idempotent() const;   // eig1 == [1], every eig0 block of size 1, other == 0
};

// (L23-1 audit R2) Every gate names where its statement lives, so a stop can be triaged: source_class is
//   "C"           a clause of [C] (source_ref = the label)          -- a bug, a refutation, or a misquoted citation;
//   "R"           a theorem of the review [R] (census_l23_classification_review.md; ref = "Thm B(2)" ...),
//                 PROVED there, not refereed into [C]                -- a bug, a refutation, or an error in [R]'s proof;
//   "LIT"         the literature (ref = the reference)              -- a bug, a refutation of the literature, or a misquote;
//   "UCT"         a universal-coefficient identity between two readings of the same model;
//   "cross-model" two independent models (C_I vs the amalgam reader) -- a disagreement is a bug;
//   "internal"    the reader's own consistency checks              -- a bug.
// G-conj gates carry the CONJ they test (class "CONJ").
struct OrbifoldGate {
    std::string name, kind;   // kind "G" or "G-conj"
    std::string source_class, source_ref;
    bool ok = true;
    std::string detail;
};
// The triage reading of a failed G gate of this source class (printed in the stop message).
std::string OrbifoldGateTriage(const std::string& source_class);

struct OrbifoldLayer {
    uint32_t q = 0;
    std::string layer;        // "whole", "sub" (H^q(V)/ell), "quot" (T^{q+1}[ell])
    JordanType jordan;        // of T_ell
};

// One computed cell (the record's content; RunOrbifoldCampaign writes it).
struct OrbifoldCell {
    uint32_t ell = 0, n = 0;
    std::string id, status;   // "ok" | "fallback" | "failed"
    bool formal = false;      // [ell = 2, n odd] ([MP] Sec. 1, Theorem O)
    // (F)
    uint32_t h[4] = {0, 0, 0, 0};
    std::vector<OrbifoldLayer> layers;
    std::map<uint32_t, bool> tp_scalar;           // p != ell -> T_p = 1 + p on every H^q, q <= 3
    uint32_t t_ell_rank[4] = {0, 0, 0, 0};
    uint32_t e_cup_h0_h2 = 0, e_cup_h1_h3 = 0;   // ranks of the induced maps
    bool e_cup_equivariant = false;
    // (Z)
    std::vector<uint32_t> ladic_type[4];          // ell-primary torsion exponents of H^q(V), descending
    uint32_t sub_dim[3] = {0, 0, 0};              // dim H^q(V)/ell, q = 1, 2
    uint32_t E = 0;                               // the (Z) reading level
    std::vector<uint32_t> ci_q1_exponents;        // torsion of H^1(V)_ell by the Smith form of delta^0 on C_I (cross-check)
    // (A) the degree-1 amalgam oracle (L23-1 audit R3): l23::h1_mod, independent of C_I
    bool amalgam_ran = false;
    uint32_t amalgam_h0 = 0, amalgam_h1 = 0, amalgam_rank_t_ell = 0, amalgam_rank_t_ell_minus_1 = 0;
    std::vector<uint32_t> amalgam_L;              // L_k = log_ell |H^1(Gamma; V/ell^k)|, k = 1..4 (ell = 2) / 1..2 (ell = 3)
    // (L)
    uint32_t rank_lambda = 0, t1 = 0, t2 = 0, t3 = 0, h0_closed = 0;
    std::vector<uint32_t> torsion1;               // the reader's exponents of H^1(V)_ell, descending
    uint32_t m_target = 0;
    // the reader's working-precision path (L23-1 audit R4): provenance, NOT in the hashed body
    uint32_t K = 0, m = 0, e_A = 0, e_1 = 0, reruns = 0;
    std::string rung;
    bool reader_ran = false, reader_ok = false;
    std::map<uint32_t, std::string> ed_tp, ed_eis;  // census convention
    std::string ed_tell;
    std::map<uint32_t, uint32_t> c_p;             // min ed_tp (cap if none finite)
    // ell = 3, the iota halves
    bool pm = false;
    uint32_t dim_plus = 0, dim_minus = 0;
    std::map<uint32_t, std::string> ed_tp_plus, ed_tp_minus, ed_eis_plus, ed_eis_minus;
    std::string ed_tell_plus, ed_tell_minus;
    bool pm_tell_halves_identical = false;
    // (L23-3) the layers of H^1(M) on the amalgam (l23::h1_layers, F_ell): (B), (U), H^1_!, and L at ell = 3; the (B) triple
    // and the H^1_! Jordan type are appended to `layers` (q = 1, "br_sub", "br_quot", "cusp").
    bool layers_ran = false;
    uint32_t am_dim_br = 0, am_dim_sub = 0, am_dim_t1 = 0, am_dim_t1_br = 0;
    uint32_t dim_h1_cusp = 0, dim_cusp_br = 0, dim_cusp_sub = 0, dim_cusp_t1 = 0;
    bool l_done = false, L_in_T1 = false, L_in_sub = false, L_stable = false;
    uint32_t dim_L = 0, dim_L_br = 0;
    // (L23-3) the level-1 q-expansion sum rules (l23::sum_rule_oracle): v_ell det(X | S_k), k = n + 2
    bool sumrules_ran = false, sumrules_resolved = false;
    uint32_t vdet_tell = 0, sumrule_K = 0, sumrule_max_invariant = 0;   // K, max_invariant: the oracle's path (not hashed)
    std::string sumrule_oracle_s;   // the oracle's wall time as printed ("0.274"), unhashed: record timings_s.sumrule_oracle (orbifold_cell/v2)
    uint32_t sumrules_skipped = 0;   // rules not gated because a divisor list is at the cap (SUMRULE_SKIPPED log lines; data)
    std::map<uint32_t, uint32_t> vdet_tp, vdet_eis, v_pn1;
    // bookkeeping
    std::vector<OrbifoldGate> gates;
    std::vector<std::string> findings, failures, log;
    uint32_t fallback_divisors = 0;
    double t_field = 0, t_ladic = 0, t_reader = 0, t_amalgam = 0, t_layers = 0, t_sumrules = 0, t_total = 0, peak_rss_mb = -1;
    uint32_t gate_failures() const;
    uint32_t conj_findings() const;
};

// (slot-23 Z2K; GPU_scout.md target 3, CPU part) THE (Z) PHASE AT ell = 2 ON THE Z/2^k LADDER.  The Smith readings
// (delta^0 over Z/2^31, delta^1 / delta^2 over Z/16) and the (U) kernel modules (Z/16) of an ell = 2 cell run on
// engine::ring2k (ENGINE_API.md Sec. 8.2: smith -> the Smith valuations, kernel_module -> the Howell form, both
// equal entrywise to engine::elim) instead of the unblocked elim::invariant_factors / elim::kernel_module.  ON by
// default; it cannot change a value (the Smith invariants and the Howell form are unique), and
// OrbifoldRing2kValidation compares whole record bodies route on vs off.  L23_RING2K=0 (read once per process;
// any value other than 0 / 1 warns and leaves the route on) is the read-only rollback; ScopedOrbifoldRing2k
// overrides it (process-wide; validation use).  ell = 3 is routed by Z3K's switch (below), never by this one.
bool OrbifoldRing2kEnabled();
// "EFFECTIVE l23_ring2k=on|off source=default|env|scoped ...": printed once per verbose orbifold campaign.
std::string OrbifoldRing2kEffectiveLine();
struct ScopedOrbifoldRing2k {
    engine::lever::EnvLever::Scoped guard_;   // S24 CL-7: the lever skeleton's RAII override
    explicit ScopedOrbifoldRing2k(bool on);
    ScopedOrbifoldRing2k(const ScopedOrbifoldRing2k&) = delete;
    ScopedOrbifoldRing2k& operator=(const ScopedOrbifoldRing2k&) = delete;
};
// Calls per route since process start (diagnostic; the validation's non-vacuity check).
struct OrbifoldRing2kCounters { uint64_t ring2k_smith = 0, ring2k_kernel = 0, engine_smith = 0, engine_kernel = 0; };
OrbifoldRing2kCounters OrbifoldRing2kCalls();
// (slot-23 Z3K; GPU_scout.md target 3, CPU part at ell = 3) THE (Z) PHASE AT ell = 3 ON THE Z/3^k RUNG.  The
// Smith readings (delta^0 over Z/3^19, delta^1 / delta^2 over Z/9) and the (U) kernel modules (Z/9) of an ell = 3 cell
// run on engine::ring3k (Ring3k.h, ENGINE_API.md Sec. 8.4: over Z/9 the vectorised small-modulus Howell pass and Smith
// by lengths, over Z/3^19 Smith by F_3 levels on the blocked LU; unit inverses through ModCtx::inv) instead of
// the scalar elim::invariant_factors / elim::kernel_module.  ON by default; it cannot change a value (the Smith
// invariants and the Howell form are unique), and OrbifoldMod3kValidation compares whole record bodies route on vs
// off.  L23_MOD3K=0 (read once per process; any value other than 0 / 1 warns and leaves the route on) is the
// read-only rollback; ScopedOrbifoldMod3k overrides it (process-wide; validation use).  ell = 2 is never routed here.
bool OrbifoldMod3kEnabled();
// "EFFECTIVE l23_mod3k=on|off source=default|env|scoped ...": printed once per verbose orbifold campaign.
std::string OrbifoldMod3kEffectiveLine();
struct ScopedOrbifoldMod3k {
    engine::lever::EnvLever::Scoped guard_;   // S24 CL-7: the lever skeleton's RAII override
    explicit ScopedOrbifoldMod3k(bool on);
    ScopedOrbifoldMod3k(const ScopedOrbifoldMod3k&) = delete;
    ScopedOrbifoldMod3k& operator=(const ScopedOrbifoldMod3k&) = delete;
};
// Calls per route since process start (diagnostic; the validation's non-vacuity check).  engine_* are the same
// process-wide engine-route counters as OrbifoldRing2kCalls() (both primes).
struct OrbifoldMod3kCounters { uint64_t ring3k_smith = 0, ring3k_kernel = 0, engine_smith = 0, engine_kernel = 0; };
OrbifoldMod3kCounters OrbifoldMod3kCalls();
// (slot-23 F3L) THE F_3 FIELD HALF OF AN ell = 3 CELL.  Two levers, one switch (integral_model::F3lRouteEnabled(),
// L23_F3L=0 the read-only rollback, read once; ScopedF3lRoute overrides it): (1) integral_model::rho_n by the O(d^2)
// column recurrence instead of the O(d^3) double sum (every caller, every ctx; the theta phase's cost), and (2) at
// ell = 3 the whole cell runs under engine::elim::ScopedTernaryF3(true), so every F_3 RREF / rank profile (the H^q
// bases, the Subquotients, the solves, the RankOfs) is TER-1's bit-sliced TritMatrix elimination.  Both are unique
// functions of their input: no value can change (OrbifoldF3lValidation compares records route on vs off).
// "EFFECTIVE l23_f3l=on|off source=default|env|scoped ...": printed once per verbose orbifold campaign.
std::string OrbifoldF3lEffectiveLine();
struct OrbifoldF3lCounters { uint64_t ternary_cells = 0, rho_recurrence = 0, rho_cubic = 0, rho_fallback = 0; };
OrbifoldF3lCounters OrbifoldF3lCalls();
// The two readings, on the LADDER route when `ladder` (engine::ring2k at ell = 2, Z2K; engine::ring3k at ell = 3, Z3K),
// else on the engine: the torsion exponents (< K, descending) of colspan(D) over Z/ell^K, and the Howell form of the
// right kernel of D over Z/ell^E.
struct OrbifoldHowell {
    engine::DenseBuf<uint32_t> H;
    std::vector<uint32_t> pivots, pivot_v;
};
std::vector<uint32_t> OrbifoldSmithTorsion(uint32_t ell, uint32_t K, const engine::DenseBuf<uint32_t>& D, bool ladder);
OrbifoldHowell OrbifoldKernelModule(uint32_t ell, uint32_t E, const engine::DenseBuf<uint32_t>& D, bool ladder);
// The reference model C_I of every cell (the coboundaries the (Z) phase reads; validation use).
const integral_model::IsotropyModel& OrbifoldReferenceModel();
// (slot-23 F3L) the census's strict lift for the Hecke prime p (the shared bundle, q_max = 3; built on first use).
const integral_model::StrictLift& OrbifoldStrictLift(uint32_t p);

// The settings block of a record (MathSettings::to_json() + the orbifold keys + the fingerprint).
std::string OrbifoldSettingsJson(const CampaignSpec& spec);
// Computes one cell (no I/O).  Never throws for a mathematical failure (status "failed", `failures`).
OrbifoldCell ComputeOrbifoldCell(const CampaignSpec& spec, uint32_t ell, uint32_t n);
// (L23-1 audit R6) The provenance block: {"commit": OC_GIT_COMMIT baked at configure time, "binary": the path of
// /proc/self/exe, "binary_sha256": its SHA-256 (computed once per process)}.  Metadata, never hashed, never branched on.
std::string OrbifoldProvenanceJson();
// The record's canonical body (everything but timings / RSS / the reader's precision path / provenance / the
// trailer) and the whole file text.
std::string OrbifoldCellBody(const CampaignSpec& spec, const OrbifoldCell& c);
std::string OrbifoldCellJson(const CampaignSpec& spec, const OrbifoldCell& c);
// The census classification of [P3] Sec. 2.4 for one cell, as a JSON object (per ideal `ord` / `nonord`).
std::string OrbifoldCensusColumnsJson(const OrbifoldCell& c);
// summary.tsv / layers.tsv of an orbifold campaign.
std::string OrbifoldSummaryHeader();
std::string OrbifoldSummaryRow(const OrbifoldCell& c, uint64_t record_fnv);
std::string OrbifoldLayersHeader();
std::string OrbifoldLayersRows(const OrbifoldCell& c);

struct OrbifoldCampaignResult {
    std::string output_dir;
    std::vector<OrbifoldCellPlan> plan;
    std::vector<OrbifoldCell> cells;              // the cells that ran, in plan order
    uint64_t ran = 0, skipped = 0, failed = 0, fallback = 0, refused = 0, findings = 0;
    uint64_t resumed_not_v2 = 0;                  // (L23-H1 audit F2) of `skipped`: complete records not in orbifold_cell/v2 (mixed-format output)
    bool stopped = false;                         // a G gate failed, or resume met a failed record: stopped ([P3] R7)
    std::string stop_reason;
    double wall_s = 0, peak_rss_mb = -1;
};
OrbifoldCampaignResult RunOrbifoldCampaign(const CampaignSpec& spec, bool verbose = true);
// The `--campaign` entry for an orbifold spec: --dry-run prints the cells; exit 0 iff nothing failed.
int RunOrbifoldCampaignCli(const CampaignSpec& spec, bool dry_run, bool verbose);

} // namespace campaign

#endif // CAMPAIGN_ORBIFOLD_H
