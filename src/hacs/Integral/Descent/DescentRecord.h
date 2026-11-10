// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

#pragma once
#ifndef INTEGRAL_DESCENT_DESCENT_RECORD_H
#define INTEGRAL_DESCENT_DESCENT_RECORD_H

// ===========================================================================
// DESCENT milestone D6 (hecke-extensions slot 24; docs/implementation/slot24/DESCENT_plan.md v0.4 Sec. 4 "D6", the carried
// L2-4; audit DESCENT_plan_audit.md m8; [X] OD-9, OD-10 (a)).  [C] = docs/latex/gl2_contract.tex, cited by label:
// thm:descentcert (the record of (1)-(4), (6)), rmk:descentcertscope, sec:descentpair (the pair's columns, which
// thm:descentcert(7)(c) assigns to D6), sec:treepair; sec:recordschema item 5 (the per-complex fields).
//
// THE RECORD `integral_descent_record/v1` of one cell (N, n) and a finite set P of Hecke primes p !| N:
//
//   object = "group"   H^q(Gamma_0(N); Sym^n Z^2), GROUP cohomology (never Bredon: OD-10's `object in {group, bredon}`;
//                      the descent computes the group object, [C] thm:descentcert(1)).
//   q_max = 3, truncated = true, q_max_source = "derived", degrees = [0 .. 3]: the per-complex fields of [C]
//                      sec:recordschema item 5 for an infinite complex reported to q_max.  q_max = 3 is DERIVED from
//                      thm:descentcert(6) ("q_max = 3 determines every degree": H^{q+2} = H^q for q >= 2 as Hecke
//                      modules), recorded as periodic_from = 2; nothing above q_max is reported, in particular no vanishing.
//   groups             H^0 .. H^3 as abelian groups (thm:descentcert(1)); the free ranks from the closed form, the
//                      ell-primary parts at 2, 3 and B(M_N) certified (D0/D1 assembly, AssembleDescentN).
//   bad_primes         B(M_N) of thm:descentcert(1): the closed list of prop:badprimesN(1) with its counts (ell >= 5);
//                      torsion_primes = {2} u {3 if n even} u B (the primes whose torsion is read).
//   K                  per torsion prime: the route and the certified precision K of the groups' run (the ell-local
//                      piece at 2, 3: the Z/ell^K ring width, W when a differential escalated; ell >= 5: the sparse
//                      cokernel's K) -- certification provenance, NOT hashed (a rollback lever may change a route or a
//                      width, never a value).
//   charpoly           per p in P: char_Z(T_p | H^1/tors) by prop:crtcharpolyN's output rule (5) (D3 CharpolyOverZ): the
//                      coefficients, `determined`, and D3's FNV digest (CharpolyZ::digest(), choice-free for certified
//                      coefficients) -- thm:descentcert(2).  (FMTMP, item 7 (l)) every coefficient beyond 2^63 also by its
//                      exact value, canonical signed hex (`big` in the JSON, one hashed body line each).
//   torsion            per (q, ell), q = 1 .. q_max at ell in {2, 3}, q = 1 at ell >= 5: the Z[T_p : p in P]-module
//                      tors H^q_(ell) by its choice-free invariants (prop:torsionhecke(1): the type, the layer
//                      polynomials, the layer blocks mod ell^{e_j}, ker / coker (T_p - c), the Fitting types; D3's
//                      canonical text, one per p, one system of generators) -- thm:descentcert(3).
//   pair               the Borel-Serre pair of sec:descentpair (D4, the tree route of sec:treepair by default):
//                      H^q, H^q(boundary), H^q_c for q = 0 .. 4, each complex with its own four fields (q_max 4,
//                      truncated, derived, [0..4]) -- the "pair columns" thm:descentcert(7)(c) assigns to D6.
//   status             "ok" | "refused" (a legitimate limit -- a cap, a precision refusal -- of some part; the part
//                      carries its reason, and the groups of a refused assembly are EMPTY, never partial).
//   settings           the mathematical settings in force (hecke::MathSettings::to_json(): `coefficient_ring = Z`, so
//                      the fingerprint is non-zero) -- part of the hash input.
//   record_fnv1a64     FNV-1a-64 (engine::FnvBytes, the one FNV) of body(): every invariant above and the settings'
//                      canonical text; NOT the K / route block, timings, RSS.  Two runs of one cell under one setting
//                      compare one word; a non-default setting changes it (a non-default
//                      mathematical setting changes the record's hash input and the campaign job id).
//
// THE RING.  BuildDescentRecord requires `coefficient_ring = Z` in force (hecke::CoefficientRingIsZ()); under F_ell or
// Z/ell^k it throws std::invalid_argument (the record is an object over Z; the setting selects it, it is never implied).
// The descent's own tools (the good prime's F_ell' pipeline, the Z/ell^K reads) run under the in-force settings with
// coefficient_ring reset to F_ell for the computation only (the golden harness's E8 precedent, reversed).
//
// CONSISTENCY (ENGINE_CHECKed: a disagreement is a defect, never a refusal): every certified torsion read has exactly
// the exponents of the group's ell-primary part (the groups' assembly and D3's read are independent Smith forms);
// the pair's bulk H^q equals the record's H^q for q <= 3.
// ===========================================================================

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "../Model/IntegralGroups.h"

namespace integral_descent {
namespace record {

using integral_model::AbelianGroupType;

constexpr const char* kSchema = "integral_descent_record/v1";
constexpr uint32_t kRecordQMax = 3;        // thm:descentcert(6): q_max = 3 determines every degree
// ([C] sec:recordschema item 7 (f), D6 audit) the pair's compact-support complex is the cone, a derived complex: the complexes it is
// derived from, reported in the JSON and on the console (outside the hashed body, so no hash and no golden line moves)
inline const std::vector<std::string> kCompactDerivedFrom = {"bulk", "boundary"};
constexpr uint32_t kRecordPairQMax = 4;    // sec:descentpair's code map (g): the pair is built to degree >= 4

struct RecordOptions {
    std::vector<uint32_t> ps;              // the Hecke primes (none dividing N); empty: DefaultRecordPrimes(N)
    bool hecke = true;                     // char_Z and the torsion Hecke modules (thm:descentcert(2), (3))
    bool pair = true;                      // the boundary pair (sec:descentpair)
    uint64_t rss_cap_bytes = 40ull << 30;  // handed to every piece (each refuses above it, rule 19)
    uint32_t threads = 0;                  // CharpolyOverZ's residue threads (0: its default)
};

struct PrimeK {
    uint32_t ell = 0;
    std::string route;                     // "tree" | "dense" (ell in {2, 3}), "sparse" | "bredon" | "shortcut" (ell >= 5 / 3)
    uint32_t K = 0, W = 0;                 // the ring width; W = the widest width read (61 when a 2-adic differential escalated)
    bool certified = false;
};
struct CharpolyEntry {
    uint32_t p = 0, r = 0;
    std::vector<std::string> coefficients;   // c_0 .. c_r: integers; "[sign,D digits]" beyond 2^63; "?" undetermined
    // (slot-24 FMTMP, [C] sec:recordschema item 7 (l)) the FAITHFUL value of every coefficient beyond 2^63: exact[i] is c_i as a
    // canonical signed hexadecimal integer ("0x1f..." / "-0x1f...": lowercase, no leading zero; ExactHex below), "" for a
    // coefficient printed as an integer in `coefficients` and for an undetermined one.  In the JSON (the entry's "big" list) and
    // in the hashed body (one "charpoly p=<p> i=<i> exact=<hex>" line each); a record without a coefficient beyond 2^63 keeps its
    // bytes and its record_fnv1a64.  Read by the console (charpoly.exact, charpoly_factored).
    std::vector<std::string> exact;
    std::vector<uint32_t> primes;          // D3's residue primes (provenance, unrecorded)
    bool determined = false;
    uint32_t undetermined = 0;
    uint64_t digest = 0;                   // D3's CharpolyZ::digest()
    bool refused = false;
    std::string refusal;
};
struct TorsionModule {
    uint32_t q = 0, ell = 0;
    std::string route;                     // "tree" | "cokernel" (provenance, unhashed)
    uint32_t K = 0;                        // the read's K (provenance, unhashed)
    std::vector<uint32_t> exponents;       // ascending (the group's ell-primary part)
    bool module_certified = false;         // 2 sigma_max <= K: the whole module (else the layer data only)
    std::vector<std::pair<uint32_t, std::string>> invariants;   // per p: D3's canonical text
    bool commute = true;
    bool refused = false;
    std::string refusal;
};
struct PairPart {
    bool ran = false;
    uint32_t q_max = kRecordPairQMax;
    std::vector<AbelianGroupType> H, Hd, Hc;   // q = 0 .. q_max: bulk, boundary, compact support (empty when refused)
    std::vector<uint32_t> primes;          // S of sec:descentpair (provenance)
    std::string route_two;                 // "tree" (the default) | "Iso2" | "IsoFree" (provenance)
    bool refused = false;
    std::string refusal;
};

struct DescentRecordZ {
    uint32_t N = 1, n = 0;
    std::vector<uint32_t> ps;
    std::string object = "group";
    uint32_t q_max = kRecordQMax;
    bool truncated = true;
    std::string q_max_source = "derived";
    std::vector<uint32_t> degrees;         // 0 .. q_max
    uint32_t periodic_from = 2;            // thm:descentcert(6)
    std::vector<AbelianGroupType> H;       // q = 0 .. q_max (EMPTY when the groups were refused)
    std::vector<std::pair<uint32_t, uint32_t>> bad_primes;   // (ell, count) of prop:badprimesN(1), ell >= 5
    std::vector<uint32_t> torsion_primes;
    std::vector<PrimeK> K;
    std::vector<CharpolyEntry> charpoly;
    std::vector<TorsionModule> torsion;
    PairPart pair;
    // (slot-24 ICP2) false: char_Z and the torsion Hecke modules were not run (RecordOptions::hecke = false); the arrays above are then
    // empty BY ABSENCE OF THE PART, not by result -- the JSON says "hecke":{"ran":false} and the hashed body carries "hecke=not_run"
    bool hecke_ran = false;
    std::string status = "ok";             // "ok" | "refused"
    std::string refusal;                   // the first refusal's line ("" when ok)
    // the settings in force (coefficient_ring = Z): hashed by their canonical text
    std::string settings_canonical, settings_json;
    uint64_t settings_fingerprint = 0;
    // provenance (unhashed)
    double t_groups = 0, t_hecke = 0, t_pair = 0, t_total = 0;
    uint64_t rss_peak = 0;

    // The hashed body: one "key=value" line per invariant, fixed order (no timings, no K / route block).
    std::string body() const;
    uint64_t digest() const;               // engine::FnvBytes(kFnvOffset64, body())
    std::string to_json() const;           // the whole record (schema kSchema), one line, fixed key order
    bool ok() const { return status == "ok"; }
};

// The first three primes not dividing N among 2, 3, 5, 7, 11, 13.
std::vector<uint32_t> DefaultRecordPrimes(uint32_t N);
// The record of the cell; requires coefficient_ring = Z in force (std::invalid_argument otherwise, and for p | N, a
// non-prime p, an empty or repeated prime list, N = 0).
DescentRecordZ BuildDescentRecord(uint32_t N, uint32_t n, const RecordOptions& opt = {});

// (slot-24 FMTMP, [C] sec:recordschema item 7 (l)) THE FAITHFUL ENCODING of a coefficient beyond 2^63.  ExactHex turns D3's
// mixed-radix digits (c^+ = sum_k d_k prod_{j<k} primes[j] in [0, P), P = prod_{k<D} primes[k]; c = c^+ - P when `negative`,
// MixedRadixNegative's convention) into the canonical signed hexadecimal text of the integer c: "-" iff c < 0, then "0x", then
// the magnitude |c| in lowercase hex without a leading zero ("0x0" for c = 0).  Word arithmetic only (Horner on a little-endian
// base-2^32 limb vector, u32 x u32 -> u64 multiply-add; one limb subtraction for the negative case): no bignum.  The text is a
// function of the integer alone (not of the residue primes), so it is choice-free and hashable.
std::string ExactHex(const std::vector<uint32_t>& digits, const std::vector<uint32_t>& primes, bool negative);
// true iff `s` is canonical ExactHex text (sign, "0x", lowercase hex digits, no leading zero, "-0x0" excluded).
bool ExactHexCanonical(const std::string& s);
// c mod Q (canonical residue; Q = 0 means 2^64) of a canonical ExactHex text, by Horner on the hex digits with D3's word mulmod --
// a decode path that shares nothing with ExactHex.  false (and *out untouched) when `s` is not canonical.
bool ExactHexModQ(const std::string& s, uint64_t Q, uint64_t* out);

// The campaign / console cost bound of one cell: psi(N) (n + 1) (the induced module's rank D).
uint64_t RecordModuleRank(uint32_t N, uint32_t n);

}  // namespace record
}  // namespace integral_descent

#endif
