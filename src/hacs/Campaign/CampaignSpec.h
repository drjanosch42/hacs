// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

#pragma once
#ifndef CAMPAIGN_SPEC_H
#define CAMPAIGN_SPEC_H

// ===========================================================================
// matrix-engine Phase 6 -- the campaign SPEC: a hand-rolled `key = value`
// description of a parameter sweep, and its expansion into shape-signature
// groups of jobs.
//
// Full documentation (grammar, semantics, worked example specs for the plan's
// three campaign targets): docs/matrix-engine/campaign.md.
//
// NO THIRD-PARTY PARSER (repo convention, plan Sec. 3): the grammar is
// deliberately line-oriented and tiny.
//
//   # or //          comment to end of line
//   ---              on a line by itself: start a new SECTION
//   key = value      a setting; `value` is a comma-separated list whose
//                    elements are either literals or `a..b step c` ranges
//                    (`step` defaults to 1).
//
// SECTIONS. The grid keys (n_sym / ell / p / r / q) are per-section, and the
// campaign is the union of the sections' cartesian products. The remaining
// keys are campaign-wide (last assignment wins, wherever it appears). One
// section is the common case; several let a campaign mix a broad low-n_sym
// sweep with a deep high-n_sym one in a single file.
//
// GRID KEYS (per section)
//   n_sym = 0..200 step 2      the symmetric-power degree n (d_V = n + 1)
//   ell   = 2,3,5,13,691       the coefficient prime
//   p     = 2,3,5              the Hecke prime (p == ell is U_ell -- same path)
//   r     = 1,2                the exponent in T_{p^r}
//   q     = 0,1,2              which cohomological degrees to REPORT
//                              (default: every degree of the model). The
//                              pipeline always computes all of them at once.
//
// CAMPAIGN KEYS
//   (hecke-extensions slot 4, WP-1 / E2a) Besides the keys below, the spec
//   accepts every MATHEMATICAL setting hecke::MathSettings registers -- today
//   `locality`, `u_ell`, `sturm`, `sturm_unavailable`, `commutant_budget`,
//   `final_mode` ([X] Sec. 1.5).  They are campaign-wide, validated by
//   HeckeSettings and nowhere else ([OD] R1), installed per job by the driver,
//   recorded in every per-job JSON, and -- unlike every ENGINE_* knob -- they
//   MAY CHANGE THE VALUE A JOB COMPUTES, so a non-default one appends
//   `_s<16 hex of the fingerprint>` to Job::id() and ` s=<16 hex>` to the
//   shape signature.  At the defaults the fingerprint is 0 and nothing is
//   appended.  See docs/matrix-engine/campaign.md Sec. 4 "Job ids".
//   (slot-9 WP-8, C5c) `coefficient_ring = Z/ell^<k>` is honoured by the JOB
//   LOOP (hecke_ring_report/v1, Campaign/CampaignRing.h, campaign.md Sec.
//   15); spec validation refuses it with `ext = true` and at any grid cell
//   with ell^k >= 2^64, before any job runs ([E] sec:ring (b4), (g)).
//
//   (hecke-extensions slot 5, WP-2 / E2 cont.) THE EXTENSION-LAYER KEYS of
//   [X] Sec. 1.5, all campaign-wide and all consulted only when `ext = true`:
//
//     ext        = true | false        run the extension engine once per
//                                      (n_sym, ell) CELL (default FALSE: with
//                                      it off today's campaign is byte-
//                                      identical, down to summary.tsv's
//                                      column set and every shape signature)
//     family     = 2,3,5,7,ell         the GOOD primes of S_0 ([E] Stage 1);
//                                      `ell` is the literal token for U_ell
//                                      and is admitted only through the
//                                      `u_ell` budget, so it is recorded and
//                                      never forces admission
//     sign       = all | plus | minus  E3: which iota-sign record the summary
//                                      reports (default `all` = both signs
//                                      AND the unsplit record)
//     ext_groups = bulk | bulk+bd | all  E4/E5: which groups (default: every
//                                      one available; `bulk` leaves the cone
//                                      unbuilt)
//
//   `u_ell` and `sturm` appear in [X]'s same block but are REGISTERED
//   MATHEMATICAL SETTINGS and are parsed by the MathSettings branch below --
//   they may change a computed value, which is why they fingerprint and the
//   four keys above do not.
//
//   name                  campaign name, echoed into index.json
//   output                output directory (created if absent)
//   resume                true|false -- skip jobs whose output file is already
//                         present AND carries the completion trailer
//   certify               full | sampled(<fraction>) | sampled(every <k>) | none
//                         (`none` = sampled with an empty sample: no external
//                          cross-check on any job; C3-C6 still always run)
//   lean                  true|false -- memory-lean mode (see campaign.md)
//   level_a               auto|always|never -- level (a) in the per-job report
//   level_a_max_entries   the `auto` size gate, in matrix entries
//   theta_rank_max_dim    skip rank(Theta_flat) above this dimension
//   model                 model descriptor name (only "sl2z_7vertex" exists)
//   jobs                  how many of a shape group's jobs run CONCURRENTLY
//                         (default 1 = the pre-Phase-4 sequential driver)
//   threads               engine worker threads PER concurrent job (default 0
//                         = inherit the process-wide pool, i.e. ENGINE_THREADS).
//                         `jobs` x `threads` is the 2-level split of Phase 6
//                         Part A; see docs/matrix-engine/campaign.md Sec. 12.
//
// SHAPE SIGNATURE AND GROUPING (the Phase 4 batching hook). Two jobs share a
// shape signature iff they agree in (n_sym, p, r, degrees, model) -- i.e.
// everything that fixes the SHAPES of the matrices and the exact Mat22 coset
// / tau tables; only `ell` varies inside a group. That is exactly axis (e) of
// 02-pipeline-consumer-audit.md Sec. 4: "several ell share the same Mat22
// tables and the same n_k, so the reductions gamma mod ell and the rho_n
// evaluations could be batched ACROSS ell into one wide kernel with a
// per-batch modulus". plan_campaign() emits the jobs already grouped and in
// group order, so the batching engineer can submit a JobGroup at a time
// without touching the driver's job loop.
// ===========================================================================

// (hecke-extensions slot 4, WP-1 / E2a) The MATHEMATICAL settings a campaign
// runs under ([X] Sec. 1.5, [OD] R1).  They are campaign-wide keys, parsed
// beside the existing ones and validated by hecke::MathSettings ONCE -- the
// engine owns the defaults and the validation, the spec only sets them.
#include "../Hecke/HeckeSettings.h"

#include <cstdint>
#include <iosfwd>
#include <string>
#include <utility>
#include <vector>

namespace campaign {

enum class CertifyMode { Full, Sampled };

// `certify`: which EXTERNAL cross-checks run. The in-process certificates
// C3-C6 are never sampled (see CampaignDriver.h).
struct CertifySpec {
    CertifyMode mode = CertifyMode::Full;
    uint32_t every_k = 1;        // select job i iff i % every_k == 0
    std::string text = "full";   // verbatim as written; echoed into the JSON
    // every_k == 0 means "select nothing" (`certify = none`).

    bool selects(uint64_t job_index) const {
        if (mode == CertifyMode::Full) return true;
        return every_k != 0 && (job_index % every_k) == 0;
    }
    std::string mode_name() const { return mode == CertifyMode::Full ? "full" : "sampled"; }
};

enum class LevelAMode { Auto, Always, Never };

// ---------------------------------------------------------------------------
// (hecke-extensions slot 5, WP-2 / E2 cont.) THE EXTENSION-LAYER KEYS of
// [X] Sec. 1.5.  `ext = false` is the default and MUST leave today's campaign
// byte-identical -- that is [X]'s own wording and this package's acceptance
// (a) -- so every field below is consulted only when `ext` is true, and
// `signature_tag()` is empty at the default.
//
// `u_ell` and `sturm`, which [X] lists in the same block, are NOT here: they
// are registered MATHEMATICAL settings and are parsed by the existing
// hecke::MathSettings branch ([OD] R1: the engine owns the defaults and the
// validation, the spec only sets).  `ext`, `family`, `sign` and `ext_groups`
// are campaign PLUMBING -- which cells the engine runs on and which objects it
// attaches -- and cannot change the value of any object the engine computes.
// ---------------------------------------------------------------------------
enum class ExtGroups { Bulk, BulkBd, All };
enum class ExtSign   { All, Plus, Minus };

struct ExtSpec {
    // `ext = true | false`.
    bool ext = false;

    // `family = 2,3,5,7,ell` -- the GOOD primes of S_0 ([E] Stage 1).  Empty
    // means the engine's own default {2, 3, 5, 7}.
    std::vector<uint64_t> good_primes;
    // The literal token `ell` in `family`: [X] Sec. 1.5 says U_ell is
    // "admitted only through the u_ell budget below", so the token is RECORDED
    // and never forces admission -- suppressing U_ell is `u_ell = off`'s job,
    // and a campaign key may not override a mathematical setting ([OD] R1).
    bool names_ell = false;

    // `ext_groups = bulk | bulk+bd | all` (default: everything available).
    // Maps onto HeckeExtDriver's `with_cone`: `bulk` leaves it off.
    ExtGroups groups = ExtGroups::All;
    // `sign = all | plus | minus` (default `all` = both signs AND the unsplit
    // record).  Maps onto `with_sign_split`; `plus` / `minus` additionally
    // select which per-sign record the summary reports.
    ExtSign sign = ExtSign::All;

    // Verbatim as written, echoed into index.json.
    std::string family_text, groups_text = "all", sign_text = "all";

    bool with_cone() const { return groups != ExtGroups::Bulk; }
    bool with_sign_split() const { return true; }   // every `sign` value splits

    // The ShapeSignature's ` ext=` component ([X] Sec. 1.5: "the shape
    // signature is extended by the family's prime set").  EMPTY when `ext` is
    // false, so every signature string a default campaign writes is unchanged.
    std::string signature_tag() const;
};

struct ShapeSignature;

// The model descriptor's own full degree list, 0 .. num_degrees()-1 -- exactly
// what a spec section WITHOUT a `q` key expands to.  Empty for an unknown
// model name.
std::vector<uint32_t> ModelDefaultDegrees(const std::string& model);

// One point of the sweep.
struct Job {
    uint32_t n_sym = 0;
    uint64_t ell = 0;
    uint64_t p = 2;
    uint32_t r = 1;
    std::vector<uint32_t> degrees;      // reported q's, ascending
    std::string model = "sl2z_7vertex";

    // Assigned by plan_campaign(), in emission order.
    uint64_t index = 0;
    bool external_certify = true;       // CertifySpec::selects(index)

    // (slot-4 WP-1 / E2a) The FNV-1a-64 fingerprint of the NON-DEFAULT
    // mathematical settings in force for this job ([X] Sec. 1.5 rule 3), 0 at
    // the defaults.  Assigned by plan_campaign() from the spec.  It is part of
    // the job's identity: a run under a different mathematical setting is a
    // DIFFERENT job, with its own output file and its own resume key, because
    // the setting may change the value the job computes.
    uint64_t settings_fingerprint = 0;

    // (slot-5 WP-2 / E2 cont.) THE EXTENSION-LAYER FIELDS.  All three are
    // assigned by plan_campaign() and are EMPTY / false unless the spec sets
    // `ext = true`, so a default campaign is byte-unchanged.
    //
    // `ext_tag` is the ShapeSignature's ` ext=` component (ExtSpec::
    // signature_tag()); it is part of the SIGNATURE and deliberately NOT of
    // Job::id(), because it cannot change any object the JOB computes -- only
    // the record the cell's OWNER attaches.
    //
    // (hecke-extensions slot 6, WP-6 / E2 close-out; owner confirmation 9)
    // It IS, separately, hashed into the ext RECORD FILE's name
    // (CampaignExt.cpp's `ExtRecordRelPath`) -- the resume-key reason: two
    // campaigns differing only in `family` must not share one record file,
    // or a `resume` under the second would silently reuse the first's
    // `S_0`-dependent record ([WP5-2] finding 3). That is a property of the
    // RECORD, not of the job, which is why the fix lives in the file name
    // and not in `Job::id()` -- confirmed here as promised above.
    std::string ext_tag;
    // True for the cell's FIRST job in plan order ([XA m14]: the record is per
    // (n_sym, ell) cell, and a job is per (n_sym, ell, p, r)).
    bool ext_owner = false;
    // The cell's operator set: the spec's `family` u {p of the cell's jobs},
    // ascending and duplicate-free.  Equal for every job of a cell.
    std::vector<uint64_t> ext_family;
    // Set by the ext pass when a cell was RESUMED: the owner's job file must
    // be rewritten even though `resume` would otherwise skip it.
    bool ext_force_rerun = false;

    // The output file's stem, the resume key and the index.json `id`.  It must
    // be INJECTIVE on everything plan_campaign() de-duplicates by, which
    // includes `degrees` -- two jobs differing only in the reported degrees are
    // distinct jobs and must not share a file (fixed 2026-09-16; see
    // docs/matrix-engine/campaign.md Sec. 4 "Job ids").
    //
    //   default degree list:      "n48_ell134217757_p2_r1"
    //   any other degree list:    "n48_ell134217757_p2_r1_q1"
    //                             "n48_ell134217757_p2_r1_q0-2"
    //
    // The "default degree list" is the model descriptor's own full list
    // 0..num_degrees()-1, i.e. exactly what a spec section without a `q` key
    // expands to, so ids are UNCHANGED for every spec that does not name `q`.
    //
    // (slot-4 WP-1 / E2a) A NON-DEFAULT mathematical setting appends, AFTER
    // any `_q` tail, `_s<the full 64-bit fingerprint, 16 lower-case hex
    // digits>`:
    //
    //   non-default settings:     "n48_ell134217757_p2_r1_s0123456789abcdef"
    //
    // The whole fingerprint and not a truncation, because the id IS the output
    // stem and the resume key and must stay injective on everything that can
    // change the job's value ([X] Sec. 1.5 rule 3; slot-4 plan exit item 4).
    // At the defaults the fingerprint is 0 and NOTHING is appended, so every
    // published file name is unchanged.  The tail order `..._r<R>[_q<tag>]
    // [_s<hex>]` keeps the injectivity argument below verbatim: `_q` may only
    // follow `_r<digits>`, and `_s` only `_r<digits>` or a `_q` tag, whose
    // characters are digits and `-` and can never look like `_s<hex>`.
    std::string id() const;
    // "n_sym=48 ell=134217757 p=2 r=1", plus " q=<list>" when the degree list
    // is not the model's default (so a log line separates them too).
    std::string to_string() const;
    // The canonical "0-1-2" / "none" tag of this job's degree list.
    std::string degree_tag() const;
    // True iff `degrees` is the model descriptor's own full degree list.  An
    // unknown model has no default, so this is false and id() always carries
    // the suffix -- injectivity is never traded for brevity.
    bool degrees_are_model_default() const;
    // The shape signature this job belongs to (single source of truth: used by
    // plan_campaign() AND by the per-job JSON's "shape_signature" field).
    ShapeSignature shape() const;
    // The GoldenHarness point string for this job's (n, ell) and (n, ell, p, r)
    // objects, so campaign hashes are directly comparable with
    // Golden/GoldenReference.inc entries.
    std::string golden_point_ne() const;
    std::string golden_point_nelpr() const;
};

// Everything that fixes the matrix shapes and the exact integer tables.
struct ShapeSignature {
    uint32_t n_sym = 0;
    uint64_t p = 2;
    uint32_t r = 1;
    std::vector<uint32_t> degrees;
    std::string model = "sl2z_7vertex";
    // (slot-4 WP-1 / E2a) 0 at the defaults, so to_string() and the ordering
    // are byte-unchanged for every campaign that sets no mathematical key.
    // Two jobs under different mathematical settings do not share a shape
    // group: a setting may change what the shared tables are used FOR.
    uint64_t settings_fingerprint = 0;
    // (slot-5 WP-2) [X] Sec. 1.5: the signature is extended by the family's
    // prime set -- but ONLY when `ext = true`, so to_string() and the ordering
    // are byte-unchanged for every campaign that does not run the ext engine.
    // It is a campaign-wide constant, so it never splits a shape group.
    std::string ext_tag;

    bool operator==(const ShapeSignature& o) const;
    bool operator<(const ShapeSignature& o) const;
    std::string to_string() const;
};

struct JobGroup {
    ShapeSignature sig;
    std::vector<Job> jobs;   // ascending in ell
};

// One section's grids (the campaign is the union of the sections' products).
struct SpecSection {
    std::vector<uint32_t> n_sym;
    std::vector<uint64_t> ell;
    std::vector<uint64_t> p;
    std::vector<uint32_t> r;
    std::vector<uint32_t> q;   // empty => every degree of the model
    // (slot-24 D6) `level = 1, 11, ...` -- the descent grid key (Gamma_0(N)), admitted ONLY under
    // `coefficient_ring = Z` (Campaign/CampaignDescent.h); empty everywhere else, so no other campaign moves.
    std::vector<uint32_t> level;
    // (slot-25 HZ-REC) `algebra = full | anemic | full,anemic` -- the grid key of a `structure = hecke` campaign
    // (Campaign/CampaignHeckeZ.h); empty everywhere else, so no other campaign moves.
    std::vector<std::string> algebra;
    uint32_t algebra_line = 0;
    uint32_t first_line = 0;   // for error messages
};

// (slot-8 WP-6) The default `ext_memory_mb` on the reference box (62 GB):
// the ten U_10007 cells sweep_full admits (n <= 18, <= ~2.05 GB each by the
// frozen U_ell model) plus their per-cell floors fit at once, with room for
// the retained program cache.
constexpr uint64_t kExtMemoryMbDefault = 40000;

// (slot-8 WP-6b, optional) The ext pass's device worker (CampaignExt.h,
// "THE DEVICE WORKER").  A SCHEDULING KNOB like ext_jobs: it changes no value,
// no id, no file name and no line of index.json / summary.tsv.
//   off   (default) no device worker: WP-6's pass, byte for byte;
//   on    one extra worker whose engine programs are compiled for the GPU
//         (a thread-local device override; ENGINE_DEVICE stays whatever it
//         is for every CPU worker) -- when a usable CUDA device exists;
//         otherwise one line is printed and WP-6's CPU-only pass runs;
//   force `on` without the availability check: the device worker runs even
//         without a device, its GPU-assigned nodes then falling back to the
//         CPU inside the executor.  The plumbing test (CheckThread under
//         ENGINE_CUDA_DISABLE=1); never needed in production.
enum class ExtGpuMode : uint8_t { Off, On, Force };

// (slot-24 ICP2) The `parts` key of a descent campaign (coefficient_ring = Z): groups only (thm:descentcert(1)); groups + char_Z + the
// torsion Hecke modules (thm:descentcert(2), (3)); all = those and the boundary pair (sec:descentpair), the default.
enum class DescentParts : uint8_t { All, GroupsHecke, Groups };
// The `parts` key's text: "all" | "groups,hecke" | "groups" (spaces around the comma tolerated); false: not a value.
bool ParseDescentParts(const std::string& text, DescentParts& out);
const char* DescentPartsText(DescentParts p);

// ---------------------------------------------------------------------------
// (hecke-extensions slot 22, L23-1) THE ORBIFOLD SECTION: `orbifold = c_i`
// turns the campaign into a campaign of `orbifold_cell/v2` records, one per
// (n_sym, ell) cell, ell in {2, 3} -- ordinary group cohomology
// H^q(SL_2(Z); Sym^n F_ell^2), q <= 3, on the isotropy-resolution model C_I and
// the ell-adic lattice of H^1(SL_2(Z); Sym^n Z^2) by the degree-1 amalgam
// reader (Orbifold/L23/AmalgamReader.h).  Plan:
// docs/implementation/slot22/census_l23_plan.md v0.3 Sec. 2 / 6 (L23-1) as
// amended by L23_master_plan.md v0.2; code Campaign/CampaignOrbifold.{h,cpp};
// user documentation docs/matrix-engine/campaign.md Sec. 16.
//
// With `orbifold` absent (or `off`) NOTHING below is consulted and the
// campaign is byte-identical to the driver without it.  The three keys
// `orbifold_primes`, `orbifold_m_target`, `orbifold_k_cap` may change a
// recorded value (which divisors are certified, which T_p enter), so a
// non-default one -- like a non-default hecke::MathSettings entry -- is part of
// the cell's identity: it enters the record's hash input and appends
// `_s<16 hex>` to the cell id (CampaignOrbifold.h, OrbifoldFingerprint()).
// ---------------------------------------------------------------------------
struct OrbifoldSpec {
    bool enabled = false;
    std::string mode_text;                       // "c_i" (the only model of this build)
    std::vector<uint32_t> primes = {2, 3, 5, 7}; // T_p on the cell (ell itself is always T_ell)
    bool primes_default = true;
    uint32_t m_target = 0;                       // 0 = auto: max(floor((n+1)/(ell+1)) + 10, 40)
    uint32_t k_cap = 0;                          // 0 = the reader's instantiated range (1024 / 312)
    uint32_t line = 0;                           // where `orbifold` was set (messages)
};

struct CampaignSpec {
    std::string name = "campaign";
    std::string output_dir = "campaign_out";
    std::string model = "sl2z_7vertex";
    bool resume = false;
    bool lean = false;
    // Phase 6 Part A -- the 2-level split.  `jobs` concurrent jobs per shape
    // group, each with its own engine pool of `threads` workers (0 = inherit
    // the process-wide ThreadPool::global(), sized by ENGINE_THREADS).
    uint32_t jobs = 1;
    uint32_t threads = 0;
    CertifySpec certify;
    LevelAMode level_a = LevelAMode::Auto;
    uint64_t level_a_max_entries = 16000000ull;  // ~ (4000)^2
    uint32_t theta_rank_max_dim = 2048;
    // (slot-4 WP-1 / E2a) The mathematical settings the campaign runs under
    // ([X] Sec. 1.5): parsed from the registered keys of hecke::MathSettings
    // beside the campaign-wide keys above, validated there and nowhere else,
    // installed per job by the driver, and recorded in every per-job JSON.
    hecke::MathSettings math;
    // (slot-5 WP-2 / E2 cont.) The extension-layer keys of [X] Sec. 1.5.
    ExtSpec ext;
    // (slot-8 WP-6) The parallel ext pass's two SCHEDULING KNOBS
    // (CampaignExt.h, "THE PARALLEL PASS"): how many cells run at once, and
    // the cap on the sum of their predicted memory.  Deliberately OUTSIDE
    // ExtSpec: they change no value, no id, no file name and no line of
    // index.json / summary.tsv -- only when a cell runs.
    uint32_t ext_jobs = 1;
    // (slot-24 ICP2) `parts = groups | groups,hecke | all` -- which parts of the integral descent record a descent cell computes
    // (coefficient_ring = Z only; Campaign/CampaignDescent.h).  The default `all` is today's campaign, id and record unchanged;
    // any other value is part of the cell id and the record marks the skipped parts "ran":false.
    DescentParts descent_parts = DescentParts::All;
    uint32_t descent_parts_line = 0;   // 0: the key was absent (a spec error outside coefficient_ring = Z when set)
    // (slot-25 HZ-REC, [C] sec:recordschema item 8 (a)) `structure = none | hecke` -- under coefficient_ring = Z, `hecke` makes the campaign
    // write integral_hecke_structure/v1 records (Campaign/CampaignHeckeZ.h) instead of descent records.  A campaign key that selects
    // the record schema, NOT a mathematical setting (Q17(v)): the default `none` is the descent campaign, byte-identical.
    bool structure_hecke = false;
    uint32_t structure_line = 0;
    // (slot-25 HZ-EMIT) `hz_cert = on | off` -- a hecke-structure campaign also writes the certificate files (WriteCertFiles) into
    // <output>/hzcert/.  An output knob: no value, id or record byte depends on it.
    bool hz_cert = false;
    uint32_t hz_cert_line = 0;
    // (slot-25 H1) `hz_g8 = off | hida` -- the record's settings value heckez.G8 ([C, sec:recordschema] item 8 (i), (j)): `off` (the
    // default: no G8 field, the default records and their fingerprint) or `hida` (H1's Hida block per maximal ideal: a non-default
    // value, so a new settings fingerprint and cell ids, never golden)
    std::string hz_g8 = "off";
    uint32_t hz_g8_line = 0;
    uint64_t ext_memory_mb = kExtMemoryMbDefault;
    // (slot-8 WP-6b) The device worker's two SCHEDULING KNOBS, outside
    // ExtSpec for the same reason: `ext_gpu` (above) and `ext_gpu_cells`, an
    // optional explicit list of (n_sym, ell) cells the device worker takes --
    // and only it -- instead of its own rule; a listed cell that is not an
    // engine cell of the pass is ignored.  Needs ext_gpu != off.
    ExtGpuMode ext_gpu = ExtGpuMode::Off;
    std::vector<std::pair<uint32_t, uint64_t>> ext_gpu_cells;
    // (slot-22 L23-1) the orbifold section (above); disabled by default.
    OrbifoldSpec orbifold;
    std::vector<SpecSection> sections;

    std::string origin;   // file name or "<string>", for messages
};

// Parses a spec. Throws std::runtime_error("<origin>:<line>: ...") on any
// syntax error, unknown key, empty grid or unknown model.
CampaignSpec ParseCampaignSpec(std::istream& in, const std::string& origin);
CampaignSpec ParseCampaignSpecFile(const std::string& path);

// (slot-22 L23-1) The orbifold section's spec-time refusals (defined in
// Campaign/CampaignOrbifold.cpp; called by ParseCampaignSpec when `orbifold`
// is enabled): throws std::runtime_error for a spec error and
// hecke::NotImplementedInThisBuild for a cell this build does not execute.
void ValidateOrbifoldSpec(const CampaignSpec& spec);
// (slot-24 D6) The descent cells' spec-time rules (defined in Campaign/CampaignDescent.cpp; called by
// ParseCampaignSpec under `coefficient_ring = Z`): throws std::runtime_error for a spec error.
void ValidateDescentSpec(const CampaignSpec& spec);
// (slot-25 HZ-REC) The hecke-structure cells' spec-time rules (Campaign/CampaignHeckeZ.cpp; called under coefficient_ring = Z with
// structure = hecke): throws std::runtime_error for a spec error.
void ValidateHeckeZSpec(const CampaignSpec& spec);

// Expands the spec into jobs, de-duplicates identical points (first occurrence
// wins), groups them by shape signature, orders the groups by signature and
// the jobs inside a group by ell, then assigns each job its campaign-wide
// index and its `external_certify` flag. Deterministic: the same spec always
// yields the same groups in the same order with the same indices.
std::vector<JobGroup> plan_campaign(const CampaignSpec& spec);

// Flattens plan_campaign()'s output back into the campaign's execution order.
std::vector<Job> flatten(const std::vector<JobGroup>& groups);

} // namespace campaign

#endif // CAMPAIGN_SPEC_H
