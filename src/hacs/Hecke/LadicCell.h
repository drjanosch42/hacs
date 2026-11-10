// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

#pragma once
#ifndef HECKE_LADIC_CELL_H
#define HECKE_LADIC_CELL_H

// ===========================================================================
// (hecke-extensions slot 11, WP-18c) THE DIGIT PIPELINE ROUTE -- one cell
// (n, ell) over Z/ell^K for ell^K >= 2^64, on the (C2) ell-adic digit
// induction of ENGINE_API.md Sec. 13.11 (engine::ladic, WP-18b).  Binding
// text: [E] v1.12.0-pre.2 sec:ring, the pre-release paragraph "v1.12.0-pre.2"
// (docs/latex/hecke_extensions.tex) -- the per-cell rule, the member list, the
// per-cell certified precision bound, the ordinary idempotent, the width
// switch; [C] obl:backend(iv) (v1.11.0-pre.2's digit-plane sentence),
// prop:certifiede.
//
// WHAT IT IS.  A NEW consumer: it instantiates NO existing class template
// (EquivariantCohomologyOnKT, HeckeOperatorOnKT, BlockMat, ...) for a new
// element type.  The shipped u32 / u64 classes are untouched, and every
// ell^k < 2^64 keeps the shipped route EXACTLY -- this file is reached only
// from the per-cell dispatch of the campaign's ring job loop and the console's
// ring lift, at ell^k >= 2^64, ell >= 5 (and from its validation, which also
// forces it below 2^64 through a TEST-ONLY hook to compare with the shipped
// route).  Everything is exact; nothing has a tolerance.
//
// WHAT A CELL COMPUTES (RunCell), at a requested output precision k:
//  1. the per-cell certified precision bound: e_max from the CERTIFIED K-mode
//     of the shipped native route at word size ([C] prop:certifiede(3)); the
//     working precision K = k + e_max; an uncertified word level is a refusal,
//     "precision insufficient", with a resumable state -- never a guess;
//  2. over Z/ell^K in digit planes: C^q_eq = kernel_module(A_Cq) (asserted
//     free), X_q (d^q in C_eq coordinates), Z^q, the (c1)-(c5) cyclic
//     decomposition of H^q, the invariant factors of d^q (the K-mode at K,
//     re-evaluated: a second, independent certificate), and the
//     universal-coefficient check of H^q(Z/ell^K) against the certified type;
//  3. for each Hecke prime p (r = 1; p == ell is U_ell): Theta^{(q)} BOTH
//     ways (fused n_k rho(s_i^{-1}) rho(g gamma_k) and route 2 n_k rho(delta_i
//     gamma_k)) from the shipped EXACT coset / tau tables, asserted equal;
//     runtime assertions (i) relations, (ii) cocycles, (iii) Theta C_eq in
//     C_eq with the exact residual; A^{(q)}(T_p) on the cyclic summands; the
//     free block of A^{(1)} and its division-free characteristic polynomial at
//     precision k;
//  4. at p == ell: the ordinary idempotent e on the free block, certified.
//
// SINGLE-THREADED (engine::ladic::Ring holds scratch buffers); CPU; no tape,
// no device (ENGINE_API.md Sec. 13.11 status).  It reads no MathSettings key
// for its values: the ring is its argument; it INSTALLS Z/ell^{K0} only around
// the shipped word-size run of step 1 (the RunE8Point precedent).
// ===========================================================================

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "HeckeSettings.h"   // NotImplementedInThisBuild
#include "../../ModArithmetic/DenseBuf.h"

namespace hecke {
namespace ladic_cell {

// --- the width switch ([E] v1.12.0-pre.2: engine-family, value-neutral) ------
// ENGINE_LADIC_WIDTH = 32 | 64 | 128 (default 64; any other text is a
// std::invalid_argument naming the variable).  It selects the digit word of
// the route; it enters no fingerprint, id or record hash.
enum class Width : uint32_t { W32 = 32, W64 = 64, W128 = 128 };
Width EngineLadicWidth();

// --- the per-cell rule ([E] v1.12.0-pre.2 sec:ring) -------------------------
constexpr uint32_t kMaxK = 4096;          // (a)'s parse cap
constexpr uint32_t kMaxBits = 8192;       // K * ceil(log2 ell) above this: refused
constexpr double kMaxPredictedBytes = 40.0 * 1024.0 * 1024.0 * 1024.0;   // rule 19's 40 GB

enum class CellRoute : uint32_t { Native = 0, Digit = 1, RefusedSmallEll = 2, RefusedBits = 3 };
const char* to_text(CellRoute r);
bool FitsU64(uint64_t ell, uint32_t k);       // ell^k < 2^64
uint32_t CeilLog2(uint64_t ell);              // ceil(log2 ell), ell >= 2
uint32_t WordLevelMax(uint64_t ell);          // K_w = max{k : ell^k < 2^64}
CellRoute ClassifyCell(uint64_t ell, uint32_t k);
// The predicted working set of a digit cell at (n, ell, K, width), bytes.
double PredictedBytes(uint32_t n, uint64_t ell, uint32_t K, Width w);
// Raises NotImplementedInThisBuild naming `who` and the cell unless the cell is
// a DIGIT cell the route executes at working precision K: ell^K >= 2^64,
// ell >= 5, K * ceil(log2 ell) <= 8192, the predicted working set under the
// cap, n > 0 even.  (The ell in {2, 3} refusal names "ell = <ell>".)
void RequireDigitCell(const std::string& who, uint32_t n, uint64_t ell, uint32_t K, Width w);

// A member of [E] sec:ring (b1)'s closure the digit route does not reach in
// slot 11 (the Bockstein, the reductions other than truncation, the boundary,
// the cone, the pairing, the ext driver, T_{p^r} r >= 2, ...): the route's own
// guard, NotImplementedInThisBuild naming the member and the cell (exit 5).
[[noreturn]] void RefuseMember(const std::string& member, uint32_t n, uint64_t ell, uint32_t k);

// "precision insufficient" -- the refused, resumable outcome of the bound.
// The R3-shaped state: kind `ladic_precision`, budget {K_w}, reached
// {r_K(d^0), r_K(d^1)} at the last word level tried, need {rank C^0_eq,
// rank C^2_eq}, complete = false.  Its continuation is a run at a larger word
// level, which this build does not have: the refusal is loud.
class PrecisionInsufficient : public NotImplementedInThisBuild {
public:
    PrecisionInsufficient(const std::string& detail, uint32_t budget_K, uint32_t r0, uint32_t r1, uint32_t need0,
                          uint32_t need1)
        : NotImplementedInThisBuild(detail), budget_K(budget_K), reached_r0(r0), reached_r1(r1), need_r0(need0),
          need_r1(need1) {}
    std::string kind = "ladic_precision";
    uint32_t budget_K = 0, reached_r0 = 0, reached_r1 = 0, need_r0 = 0, need_r1 = 0;
    bool complete = false;
};

// A residue of Z/ell^j as its j base-ell digits, low -> high (the
// width-independent serialisation of ENGINE_API.md Sec. 13.11.2).
using Residue = std::vector<uint32_t>;
// [E] v1.12.0-pre.2 (g): the signed decimal of the representative of least
// absolute value when it fits int64, else "ell-adic:[a_0,a_1,...]".
std::string ResidueText(const Residue& r, uint64_t ell);
// The residue of a (small) signed integer mod ell^j.
Residue ResidueOfInt(int64_t v, uint64_t ell, uint32_t j);

// (slot-12 WP-4) Where e_max comes from.  KMode (the DEFAULT, the slot-11
// route, unchanged): the certified word-size K-mode of a shipped run.  Prepass
// (OPT-IN): [C] v1.12.0-pre.2 prop:torsionprepass / cor:ordprecision -- e_max and
// the Z_(ell)-type of H^1 from rho_n(S), rho_n(T) alone (hecke::ordinary), before
// any cochain complex; no word-size run.  Every other computed value is the same.
enum class PrecisionSource : uint32_t { KMode = 0, Prepass = 1 };

struct Request {
    uint32_t n_sym = 0;
    uint64_t ell = 0;
    uint32_t k = 0;                        // the requested output precision, 2 <= k <= kMaxK
    std::vector<uint64_t> primes;          // Hecke primes, r = 1; p == ell is U_ell
    bool hida = true;                      // the ordinary idempotent when ell is among `primes`
    Width width = Width::W64;              // callers pass EngineLadicWidth()
    PrecisionSource precision_source = PrecisionSource::KMode;   // (slot-12 WP-4) opt-in Prepass
    bool keep_objects = false;             // fill CellRecord::objects (the oracles)
    // (slot-15 WP-5) the module consumer: the Z/ell^K Hecke-algebra route
    // (HeckeExt/HeckeRingHeckeAlgebra) needs H^1(Z/ell^K) with the T_p at ONE
    // given working precision.  module_K > 0: run at exactly K = module_K (the
    // per-cell rule is checked at module_K, not at k; k is then only the UCT
    // reading's precision, 2 <= k <= module_K; module_K <= e_max is allowed --
    // H^q(Z/ell^K) is exact at every K, only the K-mode certificate then fails).  keep_hecke: fill
    // objects.dec_exponents and objects.A only (not C, X, ...).  Both default
    // off: no existing caller's value moves.
    uint32_t module_K = 0;
    bool keep_hecke = false;
    // ---- TEST-ONLY hooks (HeckeLadicCellValidation; no production caller) ----
    bool allow_native_range = false;       // run even when ell^K < 2^64 (oracle (i))
    uint32_t fixed_K = 0;                  // run at exactly this K (the bound is still computed and recorded)
    uint32_t word_level_override = 0;      // the ONE word level tried (no escalation): forces an uncertified K0
};

struct Precision {
    uint32_t requested = 0;                // k
    uint32_t working = 0;                  // K = k + e_max (or the fixed_K hook)
    uint32_t e_max = 0;
    uint32_t word_level = 0;               // the word level that certified
    std::vector<uint32_t> word_levels_tried;
    bool word_certified = false;           // the shipped K-mode at word_level
    bool digit_certified = false;          // the stopping rule re-evaluated at K
    bool uct_match = false;                // H^q(Z/ell^K) == UCT(certified type, K), every q
    bool certified = false;                // all three (Prepass: prepass && digit && uct)
    bool prepass = false;                  // (slot-12 WP-4) e_max from the pre-pass; word_* then stay 0 / false
                                           // (slot-12 WP-11, A8) canonical_text() appends " prepass=1" iff set;
                                           // no emitter (campaign, console, golden) reaches the Prepass route --
                                           // giving it one moves no default text, but its records carry the flag.
    uint32_t digit_width = 0, digits_L = 0, digit_m = 0;
};

struct TypeQ {
    uint32_t free_rank = 0;
    std::vector<uint32_t> torsion;         // descending
};

struct Degree {
    uint32_t rank_C = 0;                   // rank C^q_eq (free)
    std::vector<uint32_t> invariant_factors_K;   // H^q(Z/ell^K), the digit run, descending
    uint32_t length_K = 0;
    std::vector<uint32_t> dK;              // invariant factors of colspan X_q over Z/ell^K
    std::vector<uint32_t> invariant_factors_k;   // H^q(Z/ell^k), the UCT reading of the certified type
    uint32_t length_k = 0;
};

struct HeckeOut {
    uint64_t p = 0;
    bool u_ell = false;                    // p == ell
    bool routes_equal = false;             // O12(a) over Z/ell^K
    uint64_t relation_checks = 0, cocycle_checks = 0, theta_eq_solves = 0;
    uint32_t rank_free = 0;                // r: the free summands of H^1(Z/ell^K)
    uint32_t precision = 0;                // K - e_max
    std::vector<Residue> free_block;       // r x r, row-major, column convention T_p b_j = sum_i A_ij b_i
    std::vector<Residue> charpoly;         // r + 1 coefficients, low -> high, monic
};

struct HidaOut {
    bool present = false;
    uint32_t precision = 0, rank = 0;      // over Z/ell^precision, r x r
    uint32_t W_exponent = 0;               // W = U^{r * precision}
    uint32_t nilpotent_dim = 0;            // a: chi_W = x^a p(x)
    std::vector<Residue> e;                // r x r, row-major
    uint32_t rank_e_mod_ell = 0, ordinary_rank_mod_ell = 0;
    bool idempotent = false, commutes = false, complement_nilpotent = false, rank_match = false;
    bool certified = false;
};

// The raw objects, serialised (LadicMat::serialise: rows, cols, then K base-ell
// digits per entry, row-major) -- for the oracles.  Filled iff keep_objects.
struct Objects {
    std::vector<std::vector<uint32_t>> C, X, Zc, H, dec_basis, dec_Gamma, dec_U;
    std::vector<std::vector<uint32_t>> dec_exponents, dec_position;
    std::vector<std::vector<std::vector<uint32_t>>> Theta, A;   // [prime][q]
};

struct CellRecord {
    uint32_t n_sym = 0;
    uint64_t ell = 0;
    Precision precision;
    std::vector<Degree> degrees;           // q = 0, 1, 2
    uint32_t r_K0 = 0, r_K1 = 0;           // at the working level K
    bool kmode_certified = false;          // the stopping rule at K
    std::vector<TypeQ> type;               // the certified Z_(ell)-type, q = 0, 1, 2
    std::vector<HeckeOut> hecke;
    HidaOut hida;
    Objects objects;
    // (S13 WP-12) wall time of the Hecke + Hida step alone (timing only: never in canonical_text, never hashed).
    double hecke_seconds = 0.0;
    // Every computed value, width-free: the cross-width comparison key.
    std::string canonical_text() const;
};

// The route.  Throws NotImplementedInThisBuild by name for a cell the route
// does not execute (RequireDigitCell), PrecisionInsufficient when no word
// level certifies, std::runtime_error / std::logic_error on a failed
// certificate (never a silently different value).
CellRecord RunCell(const Request& req);

// (slot-18 WP-19b) THE p != ell DIGIT SPLIT -- RunCell cut at the Hecke step.
// A session runs steps 1-2 (the certified precision bound and the cohomology
// over Z/ell^K, which do not depend on any Hecke prime) ONCE at construction;
// run(primes) then runs steps 3-4 for any prime list on the shared cell.
// RunCell(req) IS DigitCellSession(req).run(req.primes) -- one code path --
// so run(P) returns, for every P, exactly the record RunCell returns for the
// same Request with primes = P (canonical_text and every object; only the
// timing field hecke_seconds differs).  req.primes is not read by the
// constructor; req.hida / keep_objects / keep_hecke are applied per run.
// Construction throws exactly what RunCell throws before its Hecke step.
// Thread-safe: run() serialises on the session (the digit Ring is
// single-threaded).  Gated by HeckeLadicSplitValidation (bit-identity of the
// shared-cell records against fresh RunCell records).
class DigitCellSession {
public:
    explicit DigitCellSession(const Request& req);
    ~DigitCellSession();
    DigitCellSession(const DigitCellSession&) = delete;
    DigitCellSession& operator=(const DigitCellSession&) = delete;
    CellRecord run(const std::vector<uint64_t>& primes);
    // The record after steps 1-2 (no hecke, no hida, no objects).
    const CellRecord& base() const;
    // Wall time of steps 1-2 (timing only; never hashed).
    double cell_seconds() const;
    struct Impl;
private:
    std::unique_ptr<Impl> impl_;
};

// (slot-12 WP-4) The certified word-size K-mode alone -- RunCell's step 1 with
// its escalation policy (the starting level, then K_w), exposed for [C]
// rmk:hidacheckable (Hi1).  certified == false: no level certified (no throw).
struct WordKMode {
    bool certified = false;
    uint32_t word_level = 0;
    std::vector<uint32_t> levels_tried;
    uint32_t e_max = 0;
    std::vector<TypeQ> type;               // q = 0, 1, 2 (iff certified)
};
WordKMode CertifiedWordKMode(uint32_t n_sym, uint64_t ell);

// Serialise a native (shipped-route) matrix exactly as Objects does, at (ell, K):
// the bridge of oracle (i).
std::vector<uint32_t> SerialiseNative(const engine::DenseBuf<uint64_t>& M, uint64_t ell, uint32_t K);
std::vector<uint32_t> SerialiseNative(const engine::DenseBuf<uint32_t>& M, uint64_t ell, uint32_t K);

} // namespace ladic_cell
} // namespace hecke

#endif // HECKE_LADIC_CELL_H
