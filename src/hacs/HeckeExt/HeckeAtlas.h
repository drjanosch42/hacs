// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

#pragma once
#ifndef HECKE_EXT_HECKE_ATLAS_H
#define HECKE_EXT_HECKE_ATLAS_H

// ===========================================================================
// (hecke-extensions slot 6, WP-8 / phase E6) THE PER-ell RESIDUE ATLAS
// Sigma(ell) -- [E] `sec:atlas` (Stage 6), `cl:sigma`, `cl:sep`, `sec:field`
// tier (ii); [X] Sec. 2.2 Stage 6; the slot-6 plan Sec. 3 WP-8 and owner
// ruling 4 (2026-09-22): the slot-5 skeleton proposal plus Sigma(11) and
// Sigma(13), the canonical split field's first consumer being THE ATLAS (not
// the record), the store write-gated and outside docs/matrix-engine/results/,
// no per-rho-bar tangent pair.
// ===========================================================================
//
// WHAT IT DOES, for one ell.
//
//   1  For every even n <= ell - 1 (weight ell + 1 INCLUDED: Delta mod 11 is
//      the cuspidal weight-12 system that is neither Eisenstein-congruent nor
//      a twist of a lower weight), run HeckeExtDriver once, with the anemic
//      prime pool T = {primes <= 31} \ {ell} as its good primes (U_ell enters
//      only through the `u_ell` verdict in force, exactly as in the driver).
//   2  For every certified-local ideal collect (pi_w, {P_p}) and put it in its
//      CANONICAL SPLIT FORM ([E] `sec:field` tier (ii), `alg:split`) through
//      ONE `CanonicalField` per (ell, r) -- the reusing overload of
//      SplitInCanonicalField.  The anemic eigenvalue vector (a_q)_{q in T},
//      taken as a Galois orbit, is the system; its KEY is the
//      CanonicalLess-least vector of the orbit (coordinate-wise
//      CanonicalField::less, lexicographic in T's ascending order).  That key
//      is a function of the system alone, which is what identifies systems
//      ACROSS n -- tier (ii)'s stated purpose.
//   3  a_ell is stored per OCCURRENCE (per cell and ideal) where the `u_ell`
//      verdict admitted U_ell, as the value at the conjugate that realises the
//      key; otherwise it is `unknown` with the verdict's own reason
//      (`u_ell_over_budget`, `off`) -- NEVER guessed.  A theta-twist (j != 0)
//      carries no a_ell.
//   4  Close under the theta-twist a_q |-> q^j a_q, j in Z/(ell - 1) (on the
//      polynomial data: P_p |-> p^j P_p mod pi_w): Sigma(ell).
//   5  Greedily choose S_sep(ell) subset T separating Sigma(ell) u {chi_{a,b}}
//      as Galois orbits ([E] `cl:sep`), chi_{a,b}(q) = q^a + q^b with a + b
//      odd mod ell - 1 (the Eisenstein-type characters of [C] `cl:chi`
//      across all n).
//
// WHAT IT DOES NOT DO (the WP-8 stop line): ell = 691, the per-rho-bar
// tangent pair, ext_compare.py, identity 6 (WP-9), Stage 2(3) (WP-9), any
// record key, the first campaign.  Nothing here enters `hecke_ext/v2` and
// nothing here is golden.
//
// (slot-7 WP-5) ell = 691.  BuildAtlas is CollectAtlas (steps 1-3, the
// pipeline) followed by CloseAtlas (steps 4-5), and both were made to scale
// to Sigma(691) (~10^6 systems) WITHOUT changing a byte of output: the
// theta-twists are computed as the scaled conjugates of their base, systems
// are found through a (r, key) index, and AtlasSeparationBy counts pairs in
// groups instead of looping over them (HeckeAtlasValidation arm (L) compares
// it with the slot-6 loop).  The ell = 691 build runs outside the suite
// (`cohomology_app --atlas 691`, HeckeAtlasCli.h; atlas_build.sh), and its
// store never enters the record or golden (slot-7 owner ruling 3b (B): the
// record says `atlas_unavailable` at ell >= 691).  The per-rho-bar tangent
// pair is HeckeTangentPair.h; ext_compare.py joins the stores with a
// campaign's records.
//
// "SEPARATED" IS DECIDED EXACTLY OR CONSERVATIVELY, NEVER GUESSED.  Two orbits
// restricted to S are equal or disjoint; they are separated when their
// restricted orbits differ in size, or (same residue degree r) differ as sets
// in the one field F_{ell^r}, or (both of size 1, hence F_ell-rational)
// differ as scalars.  The one remaining case -- different r, equal restricted
// size >= 2 -- would need an embedding between canonical fields; it is counted
// as NOT separated (`undecided`), so a greedy S_sep can only be too large,
// never falsely separating.
//
// THE STORE.  `hecke_atlas/v1` JSON, byte-deterministic (no commit, no
// timing, no wall clock), written ONLY under HECKE_ATLAS_OUT=<dir>, on the
// GOLDEN_DUMP pattern; a directory under docs/matrix-engine/results/ is
// refused (slot-6 ruling 4 (i)).  `ParseAtlasStore` is its reader, and
// to_json(parse(to_json(x))) == to_json(x) is asserted by the validations.
//
// PLACEMENT.  HeckeExt/, uint32_t throughout, no hash container (rule 4), no
// Console/ Cuda/ Engine/ include (rule 2).  ENGINE_CHECK on invariants.
// ===========================================================================

#include "HeckeCanonicalField.h"
#include "HeckeIdentityCertificate.h"

#include <cstdint>
#include <functional>
#include <iosfwd>
#include <memory>
#include <string>
#include <vector>

namespace hecke_ext {

inline constexpr const char* kAtlasSchema = "hecke_atlas/v1";

// ---------------------------------------------------------------------------
// The store's data model -- exactly what `hecke_atlas/v1` serialises, so that
// the round trip is a statement about this struct.
// ---------------------------------------------------------------------------

// The a_ell of one occurrence.
struct AtlasAEll {
    bool        known = false;
    FieldElt    value;          // in F_{ell^r} of the system, iff known
    std::string reason;         // "within_budget" iff known, else the verdict's reason
};

struct AtlasOccurrence {
    uint32_t    n = 0;
    uint32_t    ideal = 0;       // the ideal's id in that cell's record
    std::string group;           // "H1", "H0", ...
    uint32_t    dim = 0;         // dim_{F_ell} of the block
    AtlasAEll   a_ell;
};

struct AtlasSystem {
    uint32_t id = 0;
    uint32_t r  = 0;                          // [F_ell(a_q : q in T) : F_ell]
    // The key: the CanonicalLess-least vector of the orbit, index-aligned
    // with AtlasStore::primes, each entry of width r.
    std::vector<FieldElt> eigenvalues;
    // "eisenstein" iff equal to some chi_{a,b}, else "cuspidal".
    std::string type;
    bool        has_chi = false;
    uint32_t    chi_a = 0, chi_b = 0;         // the least (a, b), a < b, iff has_chi
    // Sigma's construction: this system is theta^j (x) system `base`, `base`
    // being a collected (j = 0) system; the first (base, j) in closure order.
    uint32_t    base = 0;
    uint32_t    twist = 0;
    bool        collected = false;            // occurs directly in some cell
    bool        in_H1 = false;                // ... in some H^1 block
    std::vector<AtlasOccurrence> occurrences; // ascending (n, ideal)
};

struct AtlasCell {
    uint32_t n = 0;
    std::vector<uint32_t> dims;               // dim H^q, q = 0, 1, 2
    uint32_t ideals = 0;
    uint32_t local_ideals = 0;
    bool        u_ell_included = false;
    std::string u_ell_reason;
};

struct AtlasField {
    uint32_t r = 0;
    std::vector<uint32_t> C;                  // C_{ell,r}, low -> high
};

struct AtlasSeparation {
    std::vector<uint64_t> primes;             // ascending
    std::vector<uint64_t> greedy_order;       // the order the greedy chose them
    bool     separates = false;
    // (slot-7 WP-5) 64-bit: at ell = 691 |Sigma u chi| is ~10^6 and the pair
    // count ~10^12.  The JSON text of a value < 2^32 is unchanged.
    uint64_t targets = 0;                     // |Sigma u chi|
    uint64_t pairs = 0;
    uint64_t unseparated = 0;                 // pairs left unseparated
    uint64_t undecided = 0;                   // of which: the conservative case
};

struct AtlasStore {
    std::string schema = kAtlasSchema;
    uint32_t    ell = 0;
    uint32_t    n_max = 0;                    // the largest even n <= ell - 1
    std::string settings_fingerprint;         // hex, of the MathSettings in force
    std::vector<uint64_t> primes;             // the anemic pool T, ascending
    std::vector<AtlasField> fields;           // one per r present, ascending r
    std::vector<AtlasCell> cells;             // ascending n
    std::vector<AtlasSystem> systems;         // Sigma(ell), closure order
    uint32_t chi_count = 0;                   // distinct chi_{a,b} as T-systems
    uint32_t chi_in_sigma = 0;
    AtlasSeparation s_sep;                    // the greedy S_sep(ell)
    AtlasSeparation s0_anemic;                // {2,3,5,7} \ {ell}, not greedy

    std::string to_json() const;
};

// The reader.  Throws std::runtime_error on any syntactic or schema error.
AtlasStore ParseAtlasStore(const std::string& json);

// ---------------------------------------------------------------------------
// The build.
// ---------------------------------------------------------------------------

struct AtlasOptions {
    // Empty = {primes <= 31} \ {ell}.
    std::vector<uint64_t> primes;
    // The driver's Stage-4 routes.  Off by default: the atlas consumes the
    // Stage-2 character only, which is certified in Stage 2 itself; the
    // generation certificates are the record's business, not Sigma's.
    bool run_commutant = false;
    bool run_sturm = false;
    // (slot-7 WP-5) Build the cells n <= min(ell - 1, n_limit) only.  The
    // default is the whole atlas; a smaller value is a SCRATCH atlas (the
    // store's n_max says which), used by the ext_compare.py mini-campaign and
    // the development of the ell = 691 build -- never a published store.
    uint32_t n_limit = 0xffffffffu;
    // (slot-7 WP-5) One progress line per cell (n, wall, r_max, ideals, peak
    // RSS) to this stream when non-null.  Diagnostic only: nothing of it
    // enters the store.
    std::ostream* progress = nullptr;
    // (slot-8 WP-5, [KI] Sec. 4c) Called once after every cell, when the
    // cell's driver has been destroyed and its line printed.  STL only: the
    // layer never names the engine (check_heckeext_includes.sh rule 2), so a
    // caller outside HeckeExt/ injects the engine's program-cache release
    // here (main.cpp's `--atlas` branch does).  Empty = nothing is called,
    // which is every pre-WP-5 caller.  Nothing it does may enter the store.
    std::function<void()> between_cells;
};

// Diagnostic, NOT serialised: per-cell wall times (acceptance (d) of WP-8,
// the [BR6] Sec. 5 column) and the split data the validations re-derive.
struct AtlasCellTiming {
    uint32_t n = 0;
    double   total_s = 0.0;             // HeckeExtDriver::run()'s wall
    double   canonical_field_s = 0.0;   // SplitInCanonicalField over the cell
    uint32_t splits = 0;
    uint32_t r_max = 0;
};

// One collected ideal's own-field data and its canonical split, kept so that
// a validation can re-derive the twist closure on the POLYNOMIAL data.
struct AtlasSource {
    uint32_t n = 0;
    uint32_t ideal = 0;
    std::string group;
    uint32_t system = 0;                       // the system id it keyed to
    hecke::FpPoly witness_poly;                // pi_w
    std::vector<uint64_t> primes;              // S_0 of the cell, ascending
    std::vector<hecke::FpPoly> character;      // P_p, index-aligned
    CanonicalSplit split;
};

struct Atlas {
    AtlasStore store;
    std::vector<AtlasCellTiming> timings;
    std::vector<AtlasSource> sources;
    double build_s = 0.0;
};

// Runs the driver on every cell and builds Sigma(ell) and S_sep(ell).
// ell must be an odd prime >= 5 (the Glover / theta-twist setting of [E]).
// BuildAtlas == CollectAtlas followed by CloseAtlas on its store.
Atlas BuildAtlas(uint32_t ell, const AtlasOptions& opts = AtlasOptions());

// (slot-7 WP-5) The two halves of BuildAtlas, exposed so that the closure can
// be re-derived from a store without re-running the pipeline.
//
// CollectAtlas: steps 1-3 only -- the cells and the COLLECTED systems (every
// system `collected`, twist 0); chi_count, chi_in_sigma, s_sep and s0_anemic
// are left empty.
Atlas CollectAtlas(uint32_t ell, const AtlasOptions& opts = AtlasOptions());

// CloseAtlas: steps 4-5 on a store whose systems are exactly the collected
// ones (ids 0 .. c-1): the theta-twist closure, the chi counts, the greedy
// S_sep and s0_anemic.  A store that already carries a closure is first cut
// back to its collected prefix, so CloseAtlas(ParseAtlasStore(json)) re-derives
// `json` byte for byte (the integrity check of the ell = 691 store).
void CloseAtlas(AtlasStore& s);

// ---------------------------------------------------------------------------
// (slot-7 WP-3, stage 2 items 4-5; owner ruling 3b (B); [E] `sec:schema` (b))
// THE IN-PROCESS ATLAS -- the record's only atlas source.  At an ell with
// IsInProcessAtlasEll(ell) (5, 7, 11, 13) Sigma(ell) is built by BuildAtlas
// with the default options, under the mathematical settings IN FORCE, once
// per (ell, settings fingerprint) per process, behind a mutex, and shared
// read-only; no stored `hecke_atlas/v1` is ever read.  Throws at any other
// ell.  The build's own driver runs have `consult_atlas = false`, so the
// build never recurses into this cache.
// ---------------------------------------------------------------------------
std::shared_ptr<const Atlas> InProcessAtlas(uint32_t ell);
std::vector<uint64_t> InProcessAtlasSSep(uint32_t ell);   // its S_sep(ell), ascending

// ---------------------------------------------------------------------------
// Operations the validations use, exposed so that they are the ones tested.
// ---------------------------------------------------------------------------

// The orbit key of an anemic vector over F_{ell^r}: the least of its r
// Frobenius conjugates.
std::vector<FieldElt> AtlasOrbitKey(const CanonicalField& F, const std::vector<FieldElt>& v);

// theta^j on a key: a_q |-> q^j a_q, re-keyed.
std::vector<FieldElt> AtlasTwist(const CanonicalField& F, const std::vector<uint64_t>& primes,
                                 const std::vector<FieldElt>& key, uint32_t j);

// The id of the system with this (r, key), or -1.
int64_t AtlasFind(const AtlasStore& s, uint32_t r, const std::vector<FieldElt>& key);

// The separation of every pair of Sigma u chi by the primes `S` (a subset of
// store.primes); fills everything but `primes` / `greedy_order`.
AtlasSeparation AtlasSeparationBy(const AtlasStore& s, const std::vector<uint64_t>& S);

// ---------------------------------------------------------------------------
// (slot-6 WP-9) Per-ideal characters of ANY completed driver run, in the
// canonical split field -- what identity 6, the Stage 2(3) table and the
// `proof_range` comparison of HeckeExtEndToEndValidation read.
// ---------------------------------------------------------------------------

struct AtlasIdealCharacter {
    uint32_t ideal = 0;
    std::string group;                        // the ideal's OWN group, "H<q>"
    uint32_t degree = 0;                      // q
    uint32_t dim = 0;                         // dim_{F_ell} of the block
    bool     local = false;                   // certified local, split succeeded
    uint32_t r = 0;                           // residue degree (0 iff !local)
    std::vector<uint64_t> primes;             // the run's anemic primes, ascending
    // The r Frobenius conjugates of the anemic system, restricted to
    // `primes`, sorted (NOT de-duplicated).
    std::vector<std::vector<FieldElt>> orbit;
    // The anemic system generates the residue field (distinct conjugates ==
    // r).  False only where U_ell adds degree.
    bool     anemic_generates = false;
};

// Reads the UNSPLIT record's ideals of `d` (run() must have completed).
class HeckeExtDriver;
std::vector<AtlasIdealCharacter> AtlasIdealCharacters(const HeckeExtDriver& d);

// The residue-character multisets of degrees 0, 1, 2 of a run, as
// IdentityCertificate item 6 consumes them: one class per distinct (r,
// orbit), multiplicity summed as dim / r.  False, with `why`, when some
// ideal is not local or its anemic system does not generate its residue
// field (then the semisimplification is not read off the blocks).
bool AtlasResidueMultisets(const std::vector<AtlasIdealCharacter>& chars,
                           std::vector<ResidueCharacterMultiset>& out, std::string& why);

// ---------------------------------------------------------------------------
// The write gate.
// ---------------------------------------------------------------------------
// Writes <dir>/hecke_atlas_ell<ell>.json iff HECKE_ATLAS_OUT=<dir> is set and
// non-empty; returns the path written, or "" when the interlock is closed.
// Throws on a directory under docs/matrix-engine/results/ or an I/O failure.
// (slot-7 WP-5) `suffix` is inserted before ".json" (the --collect-only
// store is hecke_atlas_ell<ell>.collected.json).
std::string WriteAtlasStoreIfRequested(const AtlasStore& s, const std::string& suffix = "");

} // namespace hecke_ext

#endif // HECKE_EXT_HECKE_ATLAS_H
