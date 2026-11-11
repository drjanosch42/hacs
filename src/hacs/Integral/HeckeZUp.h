// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

#pragma once
#ifndef INTEGRAL_HECKEZ_UP_H
#define INTEGRAL_HECKEZ_UP_H

// ===========================================================================
// HeckeZUp -- slot-26 package Z1 (UP-Z (a)): U_p, p | N, over Z_(ell), ell >= 5, in the record integral_hecke_structure/v1 at full level
// N > 1 (docs/implementation/slot26/00-slot26-plan.md Sec. 2.1 "Z1"; the design UPZ_design.md v0.1 route (i) "invariant", its audit
// UPZ_design_audit.md, conditions C1-C7; report Z1_report.md).
//
// THE OPERATOR.  U_p^{(N)} = [Gamma_0(N) diag(1, p) Gamma_0(N)] for p | N: on the cokernel presentation M_N / Rel_N it is the shipped
// Heilbronn operator H_{N,p} with the admissibility filter ([C, prop:heilbronn](2), p | N stated there); on H^1 it is the ADJOINT
// U_p* = W_p U_p W_p^{-1} of the classical operator on p-old spaces ([C, prop:levelconehecke](3)(d)), the U_p of [C, def:heckeZ].
// The free-model operator on Iso (IsoHeckeN with allow_bad_primes; the level-1 lift PairLift(p) with the index-p transversal) is
// [C, prop:strictliftN](9).
//
// WHERE IT ENTERS THE RECORD (UPZ_design.md Sec. 1.3 R1-R7; only under full_N = algebra == "full" && N > 1).
//  * G1: the full rational step RationalStepFull on the operator list (G1.P's T_p ascending, then U_p ascending; item 8 (d')): the full
//    (C1) (degree d = dim M_k(Gamma_0(N))), theta, its index list G1.theta_ops (audit C2), (C2) and the discriminant primes of S_cand.
//    P itself is the anemic record's P (P_rule's (C1) is the anemic one; U_p never enlarges P).
//  * G2: at every ell >= 5 of S_cand the local read on the read set (the list) u {T_ell}, U_p on the cokernel route of the export
//    (ExportOptions::allow_bad_primes), the anemic record's generation rule (saturation, rational).  The maximal ideals and the
//    components need the Eisenstein U_p systems, which Z1 does not build: `blocked(U_p)` where the anemic rule would decompose.
//  * ell in {2, 3} (slot-26 Z2, UP-Z (b)): the same read, U_p on the tree route of the export (the free-model U_p of `prop:strictliftN` (9),
//    transported); where the search did not certify (C1), or on the non-reduced branch, no read: open(sandwich), the rule at ell >= 5.
//  * G3: the words of FullG3Rule (Z2); G4 - G7, the settings text, the schema string, record ids: unchanged.
//  * G10.contract (key of `full` records at N > 1) names the contract text (audit C6).
// Nothing at N = 1 or in an anemic record moves: the 141-record recompute (HZUP_RECOMPUTE, the Z1 merge gate, audit C3).
//
// PARALLEL DECOMPOSITION (plan Sec. 0 brief rule; UPZ_design.md Sec. 2.2): the word-prime residues of U_p run per word prime (RNS lanes,
// RationalStep's batches), the export's Heilbronn application per Mer_p slice and per column block (ExportThreads); every route computes
// canonical residues of one exact sum, so no thread count, slice or ISA body changes a value (HeckeZUpValidation arm (d)).
// ===========================================================================

#include <cstdint>
#include <string>
#include <vector>

#include "HeckeZRational.h"

namespace heckez {
namespace up {

// G10.contract of every full level-N record (audit C6; [C, sec:recordschema] item 8 (d')(vii)): the contract version whose item 8 (d') the
// record implements: the last version whose change moved a written word (not the title-page version; a signature alone does not change it).  G10 is in neither hash.
// (slot-26 HZ-EISUP, audit M3; design D5) 1.27.0-pre.1: this binary writes the words of "The components and the maximal ideals from HZ-EISUP on";
// fixed at the pre-release, before any record-writing run (Z1 wrote 1.26.0-pre.3, Z2 1.26.0-pre.9).
constexpr const char* kFullContract = "gl2_contract 1.27.0-pre.1, item 8 (d')";

// The primes dividing N, ascending (the U_p of a full level-N record).
std::vector<uint32_t> PrimesDividing(uint32_t N);
// G1.theta_ops: "T_<p>,...,U_<p>,..." for the operator list (the last nU entries are U_p).
std::string ThetaOpsText(const std::vector<uint32_t>& ops, uint32_t nU);
// The primitive theta_j of Q (in discriminants_used order) with a complete discriminant factorisation and ell certified not to divide
// it (the record's `rational` route), else -1.
int32_t RationalThetaIndex(const RationalResult& Q, uint64_t ell);

// THE STATUS RULE of G1 on a full record at N > 1 ([C] 1.26.0-pre.3 item 8 (d')(ii), (c), (d)) -- the ONE place that decides it, so a
// revised rule is a one-spot change.  Inputs: the U_p tags' verdicts and whether / how the full theta search ran.
//  * non-reduced branch (a U_p tag certifies T (x) Q non-reduced): no search; C1 = refused(non_reduced) (final), theta = not_run,
//    components = not_run, C2 = not_run, no disc fields, no theta-derived S_cand reason, no ell >= 5 read (open(sandwich)).
//  * otherwise, the search ran: C1 = certified | open(cap) (theta_cap exhausted, final); reduced_Q = true if C1 certified, else true if
//    the split-degree tag certifies it, else undecided; theta = its coefficients | open(cap); components = blocked(U_p) where (C1) holds
//    (HZ-EISUP: `certified` where e^an_k(N) = 1, the rows by HeckeZEisUp), else not_run; C2 = the full step's (C2) | partial(unfactored) beside open(cap).
//  * otherwise, the search did not run (no Z1 route served): C1, reduced_Q, theta, C2 keep blocked(U_p) (blocked(export) at an RSS
//    ceiling), and so does components ([C] 1.26.0-pre.4 W4: never not_run there); never open(cap) where no search ran.  (Z1 has no such case today: the word-prime cap refuses the whole record.)
struct FullG1Words {
    bool run_search = false;         // run RationalStepFull (and the ell >= 5 reads when (C1) is certified)
    std::string C1, reduced_Q, theta_status, components, C2;   // theta_status / C2 empty: write the value from the step
    bool disc_fields = false;        // write G1.disc.*, G1.discriminants_used, C1.deg_f
};
// Before the search (search_result == nullptr): run_search and the non-reduced words.  After it: the final words.
// (slot-26 HZ-EISUP; [C] 1.27.0-pre.1 item 8 (d')(ii)) eisup_scope = e^an_k(N) = 1 (eisup::InScope): where (C1) is certified the components are
// `certified` (HZ-EISUP writes the rows); outside the scope they keep blocked(U_p) (pending; mover HZ-EISUP (b): psi != 1).
FullG1Words FullG1Rule(bool tag_non_reduced, bool tag_split, const RationalResult* search_result, bool search_route_refused = false,
                       bool rss_ceiling = false, bool eisup_scope = false);

// THE STATUS RULE of G3 on a full record at N > 1 after Z2 ([C] 1.26.0-pre.9 item 8 (d')(ii), "The G3 words from Z2 on"; the one place):
//  * tag:   G3.C_T[U_p...].status = upper_bound_certified where that tag certifies its bound, else not_run (a U_p tag gives no bound: final);
//  * status: G3.C_T.status = upper_bound_certified where every tag bounds, else not_run;
//  * lower: G3.C_T.lower = not_run on every full record at N > 1, reduced and non-reduced branch alike: v1 reads no lower bound
//    on the U_p-tag partition (prop:heckeZcong(5)'s at-root read is stated for one rational component against the rest, not for the
//    tag pairs; [C] does not define it, so a later package would add it under a settings step).
//  * (Z2 fix round, audit S1; [C] 1.26.0-pre.10, referee W-1) these words presuppose that the tag step ran: where it did not run at some
//    p | N, G3.C_T[U_p].status of that p, G3.C_T.status and G3.C_T.lower keep blocked(U_p) (pending), never not_run, which would be
//    false-final.  As built no record reaches this (a word_prime_cap refusal refuses the whole record); the rule states it all the same.
struct FullG3Words {
    std::string tag_status, ct_status, ct_lower;
};
FullG3Words FullG3Rule(bool tag_bound, bool all_tags_bound, bool tag_ran, bool all_tags_ran);

// ---- the U_p dump (plan Z1 card, audit B2 (a); UPZ_design.md Sec. 3) ------------------------------------------------------------------
// For a full record at N > 1, one file per p | N in `dir` (which must exist): `<record id>.Up<p>.mat`, the EXPORT-Z object of U_p
// ([[tau, M], [0, F]]; ExportModuleZ2 with allow_bad_primes): a header (schema hz_up_matrix/v1, N, n, p, record id and hash, the operator
// convention, the basis rule and its FNV, r, t, the torsion moduli, F's proof), then tau (t rows, row i mod d_i), M (t rows, against the
// classes [b_j]), F (r rows, exact decimal integers, row-major), and a trailing `sha256=<hex of every preceding byte>`.  Deterministic.
// Where the object is not served (a resource refusal, a refused read) tau_M is marked in the file, or, for the free block, no file is written
// and `refused` (optional) receives "U_<p>: <reason>".  Returns the file names written (sorted).
struct UpDumpOptions {
    uint32_t threads = 0;            // the free block's lanes (0: ENGINE_THREADS, capped at 16); no byte depends on it
};
std::vector<std::string> WriteUpMatrixFiles(uint32_t N, uint32_t n, const std::string& record_id, const std::string& record_fnv,
                                            const std::string& dir, std::vector<std::string>* refused = nullptr,
                                            const UpDumpOptions& opt = {});
// true iff the text ends with `sha256=<64 hex>\n` and that hex is the sha256 of everything before that line
bool UpMatrixShaOk(const std::string& text);

// (plan Z1 card, audit S2) the descent levers as the process reads them: one line
//   EFFECTIVE HeckeZ descent HECKE_DESCENT_TREE=route=<v>,source=<default|env|env-invalid> ... (twelve tokens)
// read with the levers' own policy (exactly 0 / 1; unset or empty = the default; another value = the default, source env-invalid).
// Stdout only; no record byte depends on it.
std::string DescentEffectiveLine();
// Prints DescentEffectiveLine() once per process (the campaign's hecke_structure arm, the validations).
void PrintDescentEffectiveOnce();

}  // namespace up
}  // namespace heckez

#endif
