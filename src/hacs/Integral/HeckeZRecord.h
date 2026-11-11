// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

#pragma once
#ifndef INTEGRAL_HECKEZ_RECORD_H
#define INTEGRAL_HECKEZ_RECORD_H

// ===========================================================================
// HeckeZRecord -- slot-25 package HZ-REC: the record `integral_hecke_structure/v1` of one cell (N, n, algebra)
// ([C, sec:recordschema] item 8 (a)-(l), [C] 1.25.0-pre.3; the plan docs/implementation/slot24/heckez/04-record-census.md v0.4.1 Sec. 1
// with its slot-25 amendments Sec. 1.2 / 1.5; the card "HZ-REC" of 06-work-packages.md; report docs/implementation/slot25/HZREC_report.md,
// which lists every field with its type, vocabulary, hash class and source).  Contract labels of [C] sec:heckeZ:
// [C, prop:heckeZrat], [C, thm:heckeZcert], [C, prop:heckeZgen], [C, prop:heckeZmono], [C, prop:heckeZthrough],
// [C, prop:heckeZcong], [C, prop:heckeZeiscert], [C, prop:heckeZglue], [C, prop:heckeZsplit], [C, thm:heckeZdescent].
//
// THE MODEL.  A record is a list of FIELDS, one per structural value, each `path = value` with a VOCABULARY name: "" (free text:
// a list, a residue list, a factored text), "int", "zbig" (FMTMP's canonical signed hex, [C, sec:recordschema] item 7 (l)), "hex64",
// "sha256", "status" (the closed status list of item 8 (c) -- StatusClass), or the name of one closed enumeration (EnumValues).
// Every field carries the group it belongs to (G0 .. G10) and, where item 8 makes it conditional, its PARENT field: a field whose
// value is an exact/certified statement may not sit under a parent whose status is a bound, open, refused or pending (the
// `exact_under_bound` scan of HeckeZRecordValidation).
//
// THE HASH PARTITION (item 8 (i)).
//  * record_fnv1a64 = engine::FnvBytes(kFnvOffset64, body()): G0 (schema, key, operators, descent_ref.record_fnv1a64, status) then
//    G1 .. G8 in field order (every list already sorted by its invariant key), then the settings' canonical text and its fingerprint.
//    Every hashed field is a function of (cell, algebra, settings): no K actually used, no witness position, no dump hash, no word
//    regime, no timing, no thread count.
//  * cert_sha256 = sha256 of cert_text(): the G9 fields (the certificate layer of 07-lean-plan.md Sec. 11.1, Lean caps in its header).
//  * provenance G10 (and descent_ref.cell_sha256, the join handle): in neither hash.  The Lean status lives in the side manifest
//    lean/cells/HZ/manifest.tsv (HZ-EMIT), outside both.
//
// THE SOURCES (HZ-LOC audit F8; the include rule check_heckeext_includes.sh rule 1).  This file may not include HeckeExt/, where the
// level-1 lattice source (HeckeZLocalSource.h: ReadCellLocal) and the Bernoulli route live.  The builder therefore takes an INJECTED
// RecordSources (two function objects); HeckeExt/HeckeZRecordSource.h supplies the shipped one (DefaultRecordSources), and every
// caller -- HeckeZRecordValidation, the campaign (Campaign/CampaignHeckeZ.h, `structure = hecke`), the console verb
// `hecke_structure` (HeckeExt/RegisterHeckeZRecordVerbs.cpp) -- passes it.  A validation can inject a forged source (the tamper arms).
//
// G4 - G6 (HZ-TOR, merged e0ddc957; schema audit 2026-11-06 F3, F1).  One row group per torsion point of ReadTor (a joint
// T'-block of H^1 / ell^e with t_m > 0, keyed pt[i] in the read's block order, its residual system in G5): G4 the glue at the common
// points, G5 the relation to the free part, G6 the splitting (split(T') is written `split_on_P`, a statement about T').  q = 2 and
// generation_tors are `not_run` (no package reads them).  G8 is not in default records (the settings knob `G8`, default off).
// (slot-25 H1) under the non-default heckez.G8 = hida: G8.scope (level1 | not_run(level_N)); per ell >= 5 of S at level 1
// G8.ell=<ell>.blocks (read | not_run(no_decomposition) | not_run(forms_cap)) and, per maximal ideal m[i] of G2 (same index),
// `ordinary`; at an ordinary Eisenstein ideal through a cuspidal component `chain_key`, `g`, `L_k`, `v_Bk`, `structure`
// (fibre_LxL | regular_deg_g | open: [U, prop:u-H11], [U, prop:u-H12]), `I0_generator` (T_<q> | none_found).
// ===========================================================================

#include <cstdint>
#include <functional>
#include <string>
#include <utility>
#include <vector>

#include "HeckeZLocal.h"
#include "HeckeZRational.h"

namespace heckez {
namespace tor {
struct TorRead;
}
namespace record {

constexpr const char* kSchema = "integral_hecke_structure/v1";
constexpr const char* kCellSchema = "integral_hecke_structure_cell/v1";

// (re-audit R2) the word caps of the second route (G3.second_route.v_Bk: v_ell(B_k / k) read mod ell^(e+2), e = v_ell(g_P), by
// HeckeHidaLambda's BernoulliOverK).  The route is served iff ell^(e+5) < 2^kSecondRouteBits (the source's own u128 bound),
// ell^(e+2) + ell <= 2^kSecondRouteRingBits (ZMod) and ell^(e+2+V) + ell <= 2^kSecondRouteTangentBits (the tangent numbers,
// V = v_ell(4^(k/2) - 1)), and the read gives v < e + 2; else v_Bk = not_run and agrees = not_read.  In the settings text
// (heckez.second_route), so a wider route is a new fingerprint ([C] item 8 (j)).
constexpr uint32_t kSecondRouteBits = 128;
constexpr uint32_t kSecondRouteRingBits = 84;
constexpr uint32_t kSecondRouteTangentBits = 110;

// ---- the vocabulary ---------------------------------------------------------------------------------------------------------------
enum class StatusClass : uint8_t { Unknown, Final, Pending };
// The class of a status word of item 8 (c); Unknown for any other text.
StatusClass ClassOfStatus(const std::string& s);
// The closed list of item 8 (c) in its order, for the report and the scan.
const std::vector<std::string>& StatusWords();
// The values of a closed enumeration by name (empty: no such enumeration).  `t_ell` admits `omitted(export:<Name>)` for any
// <Name> made of [A-Za-z0-9_]: tested by ValueInVocab.
const std::vector<std::string>& EnumValues(const std::string& vocab);
// (slot-26 Z1, [C] 1.26.0-pre.3 confirm N2) the names of every closed enumeration (for the static vocabulary check)
const std::vector<std::string>& EnumVocabNames();
// true iff `value` is admissible for the vocabulary (status: a status word; int: decimal; zbig: canonical signed hex; hex64:
// 0x + 16 lowercase hex; sha256: 64 lowercase hex; an enumeration: one of its values; a "list:<enum>" vocabulary: a comma list of
// enumeration values, possibly empty; "": anything).
bool ValueInVocab(const std::string& vocab, const std::string& value);

struct Field {
    std::string group;               // "G0" .. "G10"
    std::string path;                // e.g. "G2.ell=691.generation.status"
    std::string value;
    std::string vocab;               // see ValueInVocab
    std::string parent;              // the path of the status field this one is conditional on ("" = none)
};

// ---- settings (item 8 (j): caps are record content, count-based, in the settings text) --------------------------------------------
struct RecordSettings {
    RationalSettings rat;            // theta_search / factor_budget (HZ-RAT), threads excluded from the text
    uint32_t p_rule_cap = 6;         // |P| <= 6 ([C] item 8 (j), 04 Sec. 1.5)
    uint32_t K_start_min = 2;        // the at-root read needs K >= 2; K0 = max(2, v_ell c(lambda) + 1)
    uint32_t K_steps = 3;            // the escalation step cap of the at-root read (`open(precision)` beyond it)
    uint32_t bezout_cap_d = 12;      // bezout_record_cap: d <= 12 and <= 64 KiB (v1 writes `by_generator` at every size)
    uint32_t bezout_cap_kib = 64;
    // (schema audit F5) the count cap of the irreducibility certificate of the cuspidal block; (re-audit R8, lead decision 2026-11-06)
    // 256, was 64 (degree 7 at n = 82 needed the 62nd prime); cost measured in HZREC_report.md Sec. 10
    uint32_t irred_primes = 256;
    // (schema audit F2, lead decision 2026-11-06) the G8 knob: "off" (the default) writes no G8 field and its settings line is
    // frozen byte for byte (the default records and the golden objects never move).  (slot-25 H1) "hida": H1's Hida block per
    // maximal ideal ([U, sec:u-eislambda]; 04-record-census.md Sec. 1.3 G8), a NON-default value with its own settings line, hence
    // its own fingerprint, record id and hash input, and never golden ([C] item 8 (j), (k)).  Any other value throws invalid_argument.
    std::string g8 = "off";
    // Lean caps (G9 header only; never in the settings text, item 8 (j) last sentence)
    uint32_t k1_cap_f = 8, k1_cap_b1 = 40, k7_cap_rank = 6, k7_cap_tors = 6;
    uint32_t threads = 1;            // word-prime parallelism (no value depends on it)
    // (HZ-RAT-PERF re-audit N4) VALIDATION ONLY: the word-prime cap of the U_p tags (their chi_Q(U_p) and their nested anemic
    // RationalStep), 0 (the default) = rat.word_prime_cap.  With one cap the record's own rational step refuses first on the cells
    // measured (G10.rat.primes_used above every U_p tag's chi_Q(U_p) need: (2, 6) 10 vs 3); this lets HeckeZRecordValidation reach the
    // record-level catch after UTagStep.  Never in the settings text; no value at 0.
    uint32_t utag_word_prime_cap = 0;
    // (slot-26 HZ-EISUP, audit M4) VALIDATION ONLY: at this ell (0 = never) lambda_0(U_ell) (else lambda_0 of the first U_p) is shifted by + 1
    // before the attribution at ell, which C-E7 must catch (HeckeZEisUpValidation arm (f)).  Never in the settings text; no value at 0.
    uint64_t eisup_tamper_ell = 0;
    std::string caps_text() const;   // the HeckeZ caps, one line each (the hashed settings text after the math settings)
};

// ---- the injected sources (HZ-LOC audit F8) -------------------------------------------------------------------------------------
struct LocalReadOut {                // = HeckeExt/HeckeZLocalSource.h CellLocalRead, copied by the source
    local::LatticeModule M;
    local::LocalRead L;
    bool read = false;
    std::string escalation;
};
struct RecordSources {
    // ReadCellLocal(N, n, ell, K, P, b1, opt, max_steps): the K-escalating local read (level 1 H^1 route or the EXPORT's free block)
    std::function<LocalReadOut(uint32_t, uint32_t, uint64_t, uint32_t, const std::vector<uint32_t>&, uint32_t,
                               const local::ReadOptions&, uint32_t)> local_read;
    // v_ell(B_k / k) read modulo ell^prec (-1: >= prec, or no route), level 1, ell >= 5, (ell - 1) !| k
    std::function<int32_t(uint64_t, uint32_t, uint32_t)> v_bernoulli;
    // (slot-25 H1) the G8 source under heckez.G8 = hida: the forms side of one level-1 weight (HeckeExt/HeckeHidaEisLambda.h,
    // HidaBlockAt).  Arguments: ell, k, W (precision ell^W), the operators theta is a combination of (primes; ell = T_ell, which is U
    // on the ordinary part), theta's coefficients.  Needed only when g8 = hida (ENGINE_CHECK there).
    struct HidaBlockOut {
        bool served = false;
        std::string refusal;                   // when !served: the cap (count-based) that refused
        std::vector<uint64_t> theta_ord_charpoly;   // chi(theta | e M_k) mod ell, low -> high, monic of degree rank e M_k
        uint32_t g = 0;                        // rank of the ordinary Eisenstein block of e S_k (the full residual system)
        int32_t v_Bk = -1;                     // v_ell(B_k / k) read mod ell^W (-1: >= W)
        bool lambda_one = false;               // v(G(x_k) - G(x_{k+ell-1})) = 1 (D3'), so lambda(G_a) = 1, mu = 0
        uint32_t I0_q = 0;                     // the first q != ell (ascending, to the Sturm bound) with T_q - e_q generating the block
                                               // mod ell and v(det(T_q - e_q | block)) = v_Bk; 0 = none found
        std::string k9;                        // the Lean checker K9's integers (HeckeHidaEisLambda K9Line), "" when g = 0; G9 only
        double secs = 0;                       // provenance (G10)
    };
    std::function<HidaBlockOut(uint64_t, uint32_t, uint32_t, const std::vector<uint32_t>&, const std::vector<uint32_t>&)> hida_block;
};

// ---- the record --------------------------------------------------------------------------------------------------------------------
struct DescentRef {
    std::string record_fnv1a64;      // hex64 of the descent record (parts = groups,hecke; hashed)
    std::string cell_sha256;         // the ed.1 cell file's sha256 when the cell is an ed.1 cell and the digests agree, else
                                     // "not_in_ed1" / "ed1_mismatch" (the join handle; NOT hashed)
    std::vector<uint32_t> primes;    // the descent's Hecke primes
    std::vector<std::pair<uint32_t, uint64_t>> charpoly_digests;   // (p, D3 digest) -- the ed.1 cross-check, not copied into H
    uint32_t b1 = 0;
    std::vector<uint32_t> torsion_ells;            // ell with tors H^1_(ell) != 0 or tors H^2_(ell) != 0 (G5 / G6 keys)
    std::vector<std::pair<uint32_t, uint32_t>> torsion_q_ell;   // (q, ell), q in {1, 2}, non-zero torsion
};

struct HeckeStructureRecord {
    uint32_t N = 1, n = 0;
    std::string algebra;             // "full" | "anemic"
    std::string id;                  // hz_N<N>_n<n>_<alg>_s<16 hex>
    std::string status;              // "final" | "pending" | "refused" (item 8 (c) rule)
    std::string refusal;             // when refused
    DescentRef descent;
    std::vector<Field> fields;       // G0 .. G10 in order
    std::string settings_text;       // math canonical text + caps text (hashed)
    std::string settings_fingerprint;   // hex64 FNV of settings_text
    std::string record_fnv1a64;      // hex64
    std::string cert_sha256;
    double secs = 0;
    uint64_t rss_peak = 0;

    std::string body() const;        // the hashed body (G0 .. G8 + settings)
    std::string cert_text() const;   // G9
    std::string to_json() const;     // one line, fixed key order
    // item 8 (k): golden-eligible iff final AND built under the default settings, i.e. the frozen `heckez.G8 = off` line (H1 audit
    // F1: a `hida` record is final but never golden).  Not hashed (to_json / the console only), so no record hash moves.
    bool golden_eligible() const;
    const Field* find(const std::string& path) const;
    void rehash();                   // recompute status (from the fields), record_fnv1a64, cert_sha256
};

// The record of the cell.  Requires coefficient_ring = Z in force (std::invalid_argument otherwise -- item 8 (a)), N >= 1, n even,
// algebra in {full, anemic}, `anemic` only at N > 1 (std::invalid_argument).  Runs the descent (parts = groups,hecke, the ed.1
// configuration) for descent_ref, HZ-RAT (RationalStep), at N > 1 full HZ-UTAG, and the local reads of every ell in S_cand.
HeckeStructureRecord BuildHeckeStructureRecord(uint32_t N, uint32_t n, const std::string& algebra, const RecordSettings& s,
                                               const RecordSources& src);

// The record's id: hz_N<N>_n<n>_<alg>_s<16 hex of the settings fingerprint>.
std::string RecordId(uint32_t N, uint32_t n, const std::string& algebra, const std::string& fingerprint_hex64);

// The scans of the card (HeckeZRecordValidation).
struct ScanResult {
    uint32_t values_outside_lists = 0;
    uint32_t exact_under_bound = 0;
    bool overall_rule_ok = true;     // status == pending iff some status field is pending (refused cells excepted)
    uint32_t pending_fields = 0;
    std::vector<std::string> problems;
};
ScanResult ScanRecord(const HeckeStructureRecord& R);

// One-line status signature of a record (the status table of 04 Sec. 2, compared by the validation): G1 C1 / reduced_Q / C2,
// per ell of S_cand the generation status (route), C_T, and G4-G6 summary, then the overall status.
std::string StatusSignature(const HeckeStructureRecord& R);

// ---- the ed.1 join (docs/integral_census/results/ed1/rows.tsv, integral census edition 1) ----------------------------------------
// The row of (N, n) (parts groups,hecke), or found = false.  JoinEd1 compares the descent_ref's record_fnv1a64 and every shared
// char_Z digest with the row; on agreement it sets descent.cell_sha256 to the row's cell_sha256 (the join handle, unhashed), on
// disagreement to "ed1_mismatch"; returns the number of disagreeing values (record hash + digests).
struct Ed1Row {
    bool found = false;
    std::string record_fnv1a64, cell_sha256;
    std::vector<std::pair<uint32_t, std::string>> charz;   // (p, 0x16 hex)
};
Ed1Row LookupEd1(uint32_t N, uint32_t n, const std::string& rows_tsv_path);
uint32_t JoinEd1(HeckeStructureRecord& R, const Ed1Row& row);

// ---- the campaign's cell file (schema audit F4) ---------------------------------------------------------------------------------
// The one-line cell file {"schema":kCellSchema,"id",..,"campaign",..,"record":<to_json>,"seconds",..,"rss_mib",..,"complete":true}.
std::string CellFileText(const HeckeStructureRecord& R, const std::string& campaign);
// Re-reads a cell file: the id must be `expect_id`, the settings fingerprint and the id must follow from the settings text, both
// hashes and the overall status must recompute from the fields, and ScanRecord must be clean.  On success fills *out (fields,
// hashes, status, descent_ref's two handles, seconds, RSS).  Else false with the reason in *why.
bool VerifyCellText(const std::string& text, const std::string& expect_id, HeckeStructureRecord* out, std::string* why);

// (schema audit F8) the export's refusal NAME from its text "REFUSED [<Name>] ..." ("" when none), and the record word of a refused
// export read: [Lift] with a certificate code -> refused(no_lift) (final), [ModulePrecision] -> refused(export_word) (pending), else
// blocked(export).
std::string ExportRefusalNameOf(const std::string& refusal_text);
std::string ExportWord(const std::string& refusal_text);
// (re-audit R6) the pair-lift code inside a [Lift] refusal ("the strict lift is refused [<Code>]"), "" when none.  ExportWord gives
// refused(no_lift) only for a certificate code (LiftCertificate, IsobClosure, StrictIdentity, InputTableCertificate), else
// blocked(export).
std::string LiftInnerCodeOf(const std::string& refusal_text);

// ---- HZ-EMIT: the certificate files (07-lean-plan.md Sec. 5 "Emission") ---------------------------------------------------------
// Files hzcert_N<N>_n<n>_<checker>_l<ell>.txt (key = value lines, matrices row-major decimal, deterministic) for the checkers K2
// (l0: the cell), K1 (per ell with (C3) certified by saturation) and K3 (per ell | g_P whose at-root read closes), written into
// `dir` (which must exist).  `only_ell` != 0 restricts K1 / K3 to that ell.  Returns the file names written (sorted).
std::vector<std::string> WriteCertFiles(const HeckeStructureRecord& R, const RecordSources& src, const RecordSettings& s,
                                        const std::string& dir, uint64_t only_ell = 0);

// (HZ-TOR; HZ-EISUP code audit m2) the G2 `support` word of a maximal ideal m = (m_pi) at ell for the decomposing theta (coefficients
// th_coeffs on the operators thP; an operator p | N of a full record at N > 1 is U_p) from the torsion read rd (nullptr: none).
// involved_only = false: the anemic / level-1 rule (every operator of thP); true: the full rows at N > 1 (operators with coefficient 0
// skipped), so not_run exactly where theta' involves an operator the read does not carry or a residue of degree > 1.
std::string SupportWordOfRead(const tor::TorRead* rd, uint64_t ell, const std::vector<uint32_t>& th_coeffs, const std::vector<uint32_t>& thP,
                              const std::vector<uint64_t>& m_pi, bool involved_only);

}  // namespace record
}  // namespace heckez

#endif
