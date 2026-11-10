// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

#pragma once
#ifndef INTEGRAL_HECKEZ_LOCAL_H
#define INTEGRAL_HECKEZ_LOCAL_H

// ===========================================================================
// HeckeZLocal -- slot-25 package HZ-LOC: the LOCAL READS of the Hecke structure over Z
// (docs/implementation/slot25/00-slot25-plan.md v0.2, wave 2; docs/implementation/slot24/heckez/06-work-packages.md "HZ-LOC";
// the mathematics in 01-math-core.md P 2.2, 2.4, 2.5, 2.6, 5.3, 5.4, 5.6 and Sec. 5 "Primes beyond the engine's modulus").
// Contract labels of [C] sec:heckeZ (in [C] since v1.25.0-pre.1, CITE-Z 2026-11-06):
// [C, prop:heckeZgen] = 01 P 2.2, [C, prop:heckeZmono] = P 2.4, [C, prop:heckeZlocal] = P 2.5, [C, prop:heckeZthrough] = P 2.6,
// [C, prop:heckeZcong] = P 5.2-5.4, [C, prop:heckeZeiscert] = P 5.6.
//
// INPUT. A LATTICE MODULE: L / ell^K = (Z/ell^K)^D with the matrices of T_p (p in P, column convention) on it, where L = H^1 / tors
// (the free quotient; Hecke-stable, and L (x) Z/ell^K = H^1 (x) Z/ell^K / tors (x) Z/ell^K because L is free). Two sources
// (HeckeExt/HeckeZLocalSource.h assembles them; this file never includes HeckeExt/, check_heckeext_includes.sh rule 1):
//  * level 1, ell >= 5, ell NOT a torsion prime of the cell: the shipped module route (ringalg::RunModule; u32 while ell^K < 2^32,
//    the u64 regimes of ModCtx above) -- its module is H^1(Gamma; V/ell^K) = H^1 / ell^K (no ell-torsion in H^2 at ell >= 5,
//    [C, thm:descentcert](1)), and it IS L / ell^K exactly when it is free of rank b1 (rank = b1 + the number of ell-torsion summands);
//  * otherwise (level N > 1, ell in {2, 3}, a level-1 torsion prime): the EXPORT (Integral/Descent/DescentModuleExport.h), its free
//    block -- T_p on L / ell^K for K <= K_max; K_max <= 1 for ell >= 46341 (EXPORT_report.md "For HZ-LOC / HZ-TOR", O1), so there the
//    at-root read is refused(export_word) (a SCOPE LIMIT of this package, stated in the record word).
// At a level-1 torsion prime the H^1 module is NOT L / ell^K (P2a-2): a read on it is labelled `A1_of_computed` and never yields a
// certified status (`torsion_prime_certified_from_H1` counts violations; it must be 0).
//
// WHAT IS READ, per (cell, ell) (all on the lattice module; ModCtx<uint64_t> arithmetic, every modulus ell^K < 2^63, FX-1/R-1):
//  (C3)  dim_{F_ell} A'_1(L/ell) against the a-priori d (d = dim M_k full, d^an anemic; [C, prop:heckeZrat]): equality = `certified`
//        (T'_ell = T_ell, saturated, [C, prop:heckeZgen](1)); below d = `open(sandwich)` ([C, prop:heckeZgen](3)); at level 1,
//        ell in {2, 3}, d >= 4: `refused(by_theory)` (03 B9).
//  (2.4) monogenic: theta-bar generates A'_1 (saturated, K = 1) -- tried for the HZ-RAT theta and each T_p; unsaturated: the
//        A_K route `AK_unsaturated`: the least K <= K_run with dim A'_K / ell A'_K = d, then the test in A'_K / ell A'_K
//        (valid GIVEN generation, which the record must certify elsewhere, e.g. by P 5.6).
//  (2.5) e = log_ell exp(T_ell / Z_ell[theta]) by the stopping rule (saturated case): the first K with exp Q_K < ell^K.
//  (2.6) the Eisenstein block e_m (m = (ell, theta - lambda)): its idempotent lifted to Z/ell^K (Newton 3e^2 - 2e^3), and whether
//        each component (Eis, the cusp part) passes through it: g_i(theta) e != 0 mod ell^K with K > v_ell Res(f_i, g_i).
//  (5.4) the AT-ROOT read: for an annihilator c of the cuspidal part with c(lambda) != 0 (the HZ-RAT cusp part c_theta, or locally
//        c_p = sqrt(chi_L(T_p) / (x - lambda_p)^r) mod ell^K -- the characteristic polynomial of T_p on S_k, a valid annihilator),
//        the Smith valuations a_1 <= .. <= a_r of c(T) on L / ell^K, K > v_ell c(lambda): eta_L = (+)_i Z/ell^{v - a_i}, the same for
//        every operator ([C, prop:heckeZcong], 01 P 5.4); the ell-part of g_P = min_p v_ell c_p(lambda_p) (local) and the HZ-RAT
//        certified valuation; the Eisenstein certificate CLOSES at ell iff v_ell(exp eta_L) >= v_ell(g_P) (01 P 5.3, ell-wise form):
//        then the ell-part of C_T is exactly ell^{v_ell(g_P)}.
// Refusals (record words): `refused(word_modulus)` (ell^K >= 2^63), `refused(export_word)` (export K_max < K needed, ell >= 46341),
// `blocked(export)` (the export refused), `open(precision)` (a valuation >= the run's K: re-run at a larger K), `general_case`
// (e^an_k(N) != 1: no single rational Eisenstein system, 01 P 5.7).
// ===========================================================================

#include <cstdint>
#include <string>
#include <vector>

#include "../../ModArithmetic/ModCtx.h"

namespace heckez {
namespace local {

constexpr uint64_t kWordModulusLimit = uint64_t(1) << 63;   // FX-1 / R-1: every read modulo ell^K < 2^63

// ---- Z/ell^K, K >= 1, ell^K < 2^63 (the u64 regimes of engine::ModCtx) ------------------------------------------------------------
struct Zlk {
    uint64_t ell = 0;
    uint32_t K = 0;
    uint64_t q = 0;                              // ell^K
    engine::ModCtx<uint64_t> ctx;
    // ENGINE_CHECK: ell >= 2, K >= 1, ell^K < 2^63
    static Zlk Make(uint64_t ell, uint32_t K);
    static bool Fits(uint64_t ell, uint32_t K);  // ell^K < 2^63
    uint32_t val(uint64_t x) const;              // v_ell(x mod ell^K), K for 0
    uint64_t pw(uint32_t v) const;               // ell^v mod ell^K (0 for v >= K)
    uint64_t red(uint64_t x) const { return x % q; }
    uint64_t mul(uint64_t a, uint64_t b) const { return ctx.mul(a, b); }
    uint64_t add(uint64_t a, uint64_t b) const { return ctx.add(a, b); }
    uint64_t sub(uint64_t a, uint64_t b) const { return ctx.sub(a, b); }
    uint64_t neg(uint64_t a) const { return ctx.neg(a); }
    uint64_t inv_unit(uint64_t u) const;         // u a unit mod ell
    std::string regime() const;                  // the ModCtx regime name
};

// ---- (slot-26 Z9 HZ-LOC-PAR) the route of the spans and products ----------------------------------------------------------------
// par (the default): the products of a closure pass in parallel with serial-order inserts, a semi-naive worklist (the products
// whose insert grew the span), the K = 1 closure of ReadLocal computed once, Span::reduce in an exact two-phase column-parallel
// form, MatMul row-parallel at top level.  serial: the slot-25 route byte for byte (rollback lever HECKEZ_LOCAL_PAR=0, landing
// slot 26).  No value can move (Z9_predictions.md Sec. 1; HeckeZLocalParValidation).  threads = 0: the consumer pool's size.
struct ParRoute {
    bool par = true;
    uint32_t threads = 0;
    bool force = false;    // validation only: the parallel forms also below their size thresholds (small cells exercise them)
};
ParRoute CurrentParRoute();        // the scoped override if set, else the lever's default
const char* DefaultParRouteName(); // "par" | "serial(HECKEZ_LOCAL_PAR=0)"
struct ParCallCounts { uint64_t reduce_par = 0, matmul_par = 0; };
ParCallCounts ParCalls();          // process-wide counts of parallel Span::reduce / MatMul invocations (gate evidence)
class ScopedParRoute {             // validation use, from one thread, not nested across threads
public:
    explicit ScopedParRoute(ParRoute r);
    ~ScopedParRoute();
    ScopedParRoute(const ScopedParRoute&) = delete;
    ScopedParRoute& operator=(const ScopedParRoute&) = delete;
private:
    bool had_;
    ParRoute saved_;
};

using Mat = std::vector<uint64_t>;               // D x D, row-major, entries in [0, ell^K)
Mat MatIdentity(size_t D);
Mat MatMul(const Zlk& R, const Mat& A, const Mat& B, size_t D);
Mat MatAdd(const Zlk& R, const Mat& A, const Mat& B);
Mat MatScale(const Zlk& R, const Mat& A, uint64_t s);
Mat MatMinusScalar(const Zlk& R, const Mat& A, uint64_t s, size_t D);   // A - s I
// p(A) for p low -> high (Horner)
Mat MatPolyEval(const Zlk& R, const std::vector<uint64_t>& p, const Mat& A, size_t D);
bool MatIsZero(const Mat& A);

// ---- polynomials over Z/ell^K, low -> high --------------------------------------------------------------------------------------
using Poly = std::vector<uint64_t>;
// det(x I - A), division-free (Berkowitz), monic of degree D
Poly Charpoly(const Zlk& R, const Mat& A, size_t D);
// p / (x - a)^r exactly; false iff a remainder is non-zero (not exact mod ell^K)
bool DivideByRoot(const Zlk& R, Poly p, uint64_t a, uint32_t r, Poly* out);
// the monic c with c^2 = p (p monic of even degree), ell odd; false iff none (c^2 != p)
bool MonicSqrt(const Zlk& R, const Poly& p, Poly* out);
uint64_t PolyEval(const Zlk& R, const Poly& p, uint64_t x);
// residue of a canonical signed hex integer ("0x..", "-0x..") modulo ell^K
uint64_t HexResidue(const Zlk& R, const std::string& hex);

// ---- Smith valuations and Howell spans over Z/ell^K -----------------------------------------------------------------------------
// The valuations s_j < K of the non-zero Smith invariants of an r x c matrix (row-major), ascending.
std::vector<uint32_t> SmithValuations(const Zlk& R, std::vector<uint64_t> A, size_t r, size_t c);

// A submodule of (Z/ell^K)^m in Howell form: rows with pairwise distinct pivot columns, pivot entry exactly ell^{v_i}, the
// annihilator multiples ell^{K - v_i} row_i inserted, so that reduction decides membership and |M| = ell^{sum (K - v_i)}.
class Span {
public:
    Span(const Zlk& R, size_t m);
    bool insert(std::vector<uint64_t> v);           // true iff the span grew
    bool contains(std::vector<uint64_t> v) const;
    uint32_t length() const;                        // log_ell |M| = sum (K - v_i)
    uint32_t generators() const;                    // dim_F_ell M / ell M = #Smith invariants < K of the rows (re-derived)
    uint32_t length_by_smith() const;               // sum (K - s_j) over the rows' Smith invariants (the cross-check)
    std::vector<std::vector<uint64_t>> rows() const;
    size_t size() const { return rows_.size(); }
private:
    struct Row { size_t piv; uint32_t v; std::vector<uint64_t> a; };
    bool reduce(std::vector<uint64_t>& v, size_t* conflict) const;   // conflict: index of the row whose pivot blocks v
    bool reduce_serial(std::vector<uint64_t>& v, size_t* conflict) const;
    bool reduce_par(std::vector<uint64_t>& v, size_t* conflict, uint32_t threads) const;   // bit-identical to reduce_serial
    const Zlk* R_;
    size_t m_;
    std::vector<Row> rows_;                         // ascending pivot
};

// The Z/ell^K-algebra generated by the matrices `gens` (D x D) inside End((Z/ell^K)^D): the span of {1} closed under right
// multiplication by every generator (commuting generators: this is Z/ell^K[gens]).
Span AlgebraClosure(const Zlk& R, const std::vector<Mat>& gens, size_t D);
// Z/ell^K[theta]: the span of 1, theta, .., theta^{D-1} (Cayley--Hamilton bounds the degree).
Span PowerSpan(const Zlk& R, const Mat& theta, size_t D);

// ---- the lattice module ---------------------------------------------------------------------------------------------------------
struct LatticeModule {
    uint32_t N = 1, n = 0;
    uint64_t ell = 0;
    uint32_t K = 0;                  // the ring of the matrices: Z/ell^K
    uint32_t D = 0;                  // rank L = b1
    std::vector<uint32_t> P;         // the operators, T_p for p in P
    std::vector<Mat> T;              // per p, D x D, column convention, mod ell^K
    std::string source;              // "h1_level1" | "export" | "h1_level1_torsion(A1_of_computed)"
    bool is_lattice = true;          // false: the module is H^1 / ell^K at a torsion prime (A1_of_computed only)
    uint32_t h1_rank = 0;            // rank of the H^1 module the source read (= D when it is the lattice)
    uint32_t K_max = 0;              // the largest K the source could serve (export: K_max; H^1 route: the run's K)
    std::string regime;              // ModCtx regime of Z/ell^K
    // (HZ-LOC audit F1) the T_ell of the read set: "in_P" | "added" | "omitted(cap:ell>23170)" | "omitted(ell|N:U_ell_not_exported)"
    // | "omitted(export:<refusal>)"
    std::string t_ell;
    // (HZ-LOC audit F3) the word the module route actually ran on: "u32" | "u64" | "digit" | "hida" (the H^1 route,
    // ringalg::RunModule's OpModule::word) or "export" (the EXPORT's free block)
    std::string word;
    bool ok = false;
    std::string status;              // "ok" | "refused(word_modulus)" | "refused(export_word)" | "blocked(export)" | ...
    std::string detail;
};

// (HZ-LOC audit F1, lead decision 2026-11-06) the read set at ell: P, plus T_ell when ell does not divide N, is not in P and
// ell <= kTEllCap (count-based: the coset count of T_ell is ell + 1; 23170 is the narrow U_ell tier of [C, lem:bounds]).  At ell | N
// the design defines no U_ell lift (the EXPORT refuses p | N), so P is unchanged there.  T_ell is APPENDED (HZ-RAT's theta stays
// aligned with the first |P| operators).  *t_ell receives the LatticeModule::t_ell word.
constexpr uint64_t kTEllCap = 23170;
std::vector<uint32_t> ReadPrimes(uint32_t N, uint64_t ell, const std::vector<uint32_t>& P, std::string* t_ell);

// The export's free block at K (level N, ell in {2, 3}, or a level-1 torsion prime). K > K_max -> refused(export_word) when
// ell >= 46341 (the word limit of the route) else open(precision); the export's own refusal -> blocked(export).
// (S26 Z1) allow_bad_primes: p | N in P is U_p (ExportOptions::allow_bad_primes, the cokernel route at ell >= 5); default false.
LatticeModule LatticeFromExport(uint32_t N, uint32_t n, uint64_t ell, uint32_t K, const std::vector<uint32_t>& P,
                                bool allow_bad_primes = false);
// Reduce a lattice module to K' <= K (the matrices mod ell^{K'}).
LatticeModule ReduceLattice(const LatticeModule& M, uint32_t K);

// ---- the rational input (from HZ-RAT where it ran) ------------------------------------------------------------------------------
struct RationalInput {
    bool present = false;
    std::vector<uint32_t> theta;     // c_p over P (HZ-RAT's primitive theta), empty: none
    std::string lambda_hex;          // lambda_theta
    std::vector<std::string> c_hex;  // the cusp part c_theta (low -> high)
    int32_t v_c_lambda = -1;         // v_ell(c_theta(lambda_theta)) from the certified factorisation (-1: ell not certified there)
    int32_t v_gP = -1;               // v_ell(g_P) from the certified factorisation (-1: unknown)
    bool gP_complete = false;        // g_P fully factored (`upper_bound_certified`)
};

struct ReadOptions {
    int64_t d_target = 0;            // d (full) or d^an (anemic), a-priori [C, prop:heckeZrat]
    uint32_t r_eis = 1;              // rank L_S = e_k (the multiplicity of lambda in chi_L(T_p))
    bool eis_rational = true;        // e^an_k(N) = 1
    uint32_t k_weight = 0;           // k = n + 2 (lambda_p = 1 + p^{k-1})
    bool full_algebra = true;        // N = 1
    RationalInput rat;
    // (slot-26 Z1) a prime p | N in the read set is U_p (the full level-N record of HeckeZUp.h); passed to the export.  Default false.
    bool allow_bad_primes = false;
};

struct RootRead {
    std::string op;                  // "theta" | "T_p"
    std::string annihilator;         // "hz_rat_cusp_part" | "local_charpoly_S_k" | "local_chi/(x-l)^r"
    int32_t v_c_lambda = -1;         // v_ell(c(lambda)) read mod ell^K (-1: >= K)
    std::vector<uint32_t> a;         // Smith valuations of c(T) on L / ell^K
    std::vector<uint32_t> eta;       // the exponents v - a_i of eta_L (empty when unread)
    std::string status;              // "ok" | "open(precision)" | "rank_mismatch" | "not_exact"
};

struct LocalRead {
    uint32_t N = 1, n = 0;
    uint64_t ell = 0;
    uint32_t K = 0;
    std::string source;
    bool is_lattice = true;
    uint32_t D = 0;
    int64_t d = 0;
    std::vector<uint32_t> P;
    std::string regime;
    std::string t_ell;               // LatticeModule::t_ell
    std::string word;                // LatticeModule::word
    // (C3)
    uint32_t dim_A1 = 0;
    std::string c3;                  // "certified" | "open(sandwich)" | "refused(by_theory)" | "A1_of_computed"
    // (2.4) / (2.5)
    std::string mono;                // "theta" | "T_<p>" | "none" | "open(sandwich)" | "AK_unsaturated(K=..):T_<p>|none" | "open(cap)"
    int32_t e = -1;                  // the stopping-rule exponent (saturated); -1 unread
    std::string e_op;                // the theta e refers to (the first candidate: HZ-RAT's theta, else T_{P[0]})
    std::string p26_op;              // the theta of the (2.6) block
    std::string e_status;            // "stopped(K=..)" | "open(precision)" | "not_saturated" | ...
    uint32_t K_gen = 0;              // AK route: the K with dim A'_K / ell A'_K = d
    // (2.6) on the Eisenstein block
    std::string p26;                 // "eis+cusp" | "eis" | "cusp" | "none" | "open(precision)" | "not_run"
    uint32_t eis_block_rank = 0;     // rank_F_ell of the idempotent mod ell
    // (5.4)
    std::vector<RootRead> roots;
    int32_t v_gP_local = -1;         // min_p v_ell c_p(lambda_p) (local; -1 unread)
    int32_t v_gP = -1;               // the valuation the certificate uses (HZ-RAT certified, else local; with T_ell in P the
                                     // min with v_ell c_ell(lambda_ell): v_ell(g_{P u {ell}}), audit F1)
    std::string v_gP_source;
    int32_t v_eta = -1;              // v_ell(exp eta_L)
    std::vector<uint32_t> eta;       // the exponents of eta_L
    std::string root;                // "closes" | "upper_bound_certified" | "open(precision)" | "general_case" | "inconsistent" |
                                     // "not_run" | "A1_of_computed"
    std::string C_T;                 // "ell^v exact" | "ell^<=v upper_bound" | ""
    bool certified_any() const;      // some certified status (c3 certified, root closes, mono found on the saturated path)
    std::string text() const;        // canonical text (the digest input)
    uint64_t digest() const;
};

LocalRead ReadLocal(const LatticeModule& M, const ReadOptions& opt);

}  // namespace local
}  // namespace heckez

#endif
