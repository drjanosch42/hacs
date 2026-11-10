// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

#pragma once
#ifndef INTEGRAL_DESCENT_DESCENT_HECKE_H
#define INTEGRAL_DESCENT_DESCENT_HECKE_H

// ===========================================================================
// DESCENT milestone D3 (hecke-extensions slot 24; docs/implementation/slot24/DESCENT_plan.md v0.4 Sec. 4 "D3").
// [C] = docs/latex/gl2_contract.tex (v1.23.0-pre.31), cited by label: sec:descenthecke -- prop:strictliftN, prop:crtcharpolyN,
// prop:torsionhecke, thm:descentcert, rmk:descentcertscope, and its code map with the obligations (a)-(d) of v1.23.0-pre.25
// (C-D4_referee.md, Author's response); sec:isomodeltree (prop:isomodeltree(3), (4), (6): Hecke by transport); prop:descentmaps
// (2)(g) (the precision rule); prop:heilbronn (the shipped operator on the cokernel presentation).  The owner's rulings: residues and a
// certified bound, no big integer [R-Q2]; Hecke output as INVARIANTS only, no integer Hecke matrix (D8) [R-Q3]; T_p for p !| N only,
// U_p (U_2, U_3 included) deferred [R-Q4] (since S26 Z1 / Z2 served on request, allow_bad_primes; the default still refuses); the tree is the 2-adic route [R-GT].  At ell in {2, 3} Hecke goes ONLY through the transport
// of the free model's operator (the C-D4 referee's M2): the shipped Bredon tau is never read there.
//
// WHAT IS COMPUTED, for A = Gamma_0(N), V = Sym^n Z^2, M_N = Ind_A^Gamma V, T_p = [A diag(1,p) A], p !| N prime:
//
//  (A) char_Z(T_p | H^1(A; V)/tors) (prop:crtcharpolyN): residues at word primes ell' (descending from 2^30; every one > n + 1, so
//      s_ell'(M_N) = 0, (2)) of the SHIPPED level-N operator on the O-piece -- H_{N,p} on the cokernel presentation M_N / Rel_N over
//      F_ell' (hecke::level::CokernelHeckeMatrices, prop:heilbronn(2)) -- no 2- or 3-adic input; the certified bound
//      b_{p,i} = r_N + i (n+1) bitlen(p) of (3); recovery by Garner's mixed radix in 64-bit words (4), the sign by the digits of
//      (P-1)/2, the residues mod Q in {2^64, 3^40, ell^K} by Horner with a word mulmod (shift-and-add, never a double word); the
//      output rule (5): certified and |c_i| < 2^63: the integer; certified and larger: sign, digits, primes, bits; not certified: the
//      residues and the bound, flagged undetermined.  A held-out prime is a consistency check, not a certificate.
//  (B) the level-N operator T~^{(N)}_{Iso,p} of prop:strictliftN(3) on C^q_I(M_N (x) R) -- eq:levelcorr with the bulk strict lift
//      tau_Iso of (2), here D4's certified lift (pair::PairLift, StrictLift::certify(): (T1)-(T3) over Z, the strict identity (S) against
//      tau_X = tau^s, built once per p on {s_j e_c}, independent of r_t and N: code-map obligations (b), (c)); applied SPARSELY (block
//      terms rho_n(v_i s_j^{-1} g gamma w^{-1}) on the coset blocks), never densified at scale.
//  (C) its transport to the ell-local tree model (prop:strictliftN(5), prop:isomodeltree(6)): T'_{T,p} = Psi_T^* T~ Phi_T^* on
//      C^._{T,ell}(M_N) = Hom(Iso^(ell)_T, M_N) at ell = 2 AND ell = 3, every N (obligation (a): no branch on nu_3(N)); Phi_T, Psi_T of
//      prop:isomodeltree(4)(d) (IntegralTreeComparison), the tree cochains in J-coordinates (bases of M^{Q_x} by Nakayama).
//  (D) the torsion read (prop:torsionhecke(3)): a certified Smith form L d^{q-1} R = Sigma over Z/ell^K with left transform, the
//      operator applied to the |J| columns of L^{-1} only, Phi~_ij = (L F x~_j)_i mod ell^{sigma_i}; the module (and every invariant of
//      (1)) iff 2 sigma_max <= K, the layer polynomials P_j and the Fitting types at every certified K; the image checked to be torsion
//      (zero free coordinates mod ell^{K'}) and a cocycle (d^q F x~ = 0 mod ell^{K'}), K' = K - sigma_max.  The choice-free invariants
//      of (1): the layer polynomials P_j, the layer blocks' polynomials mod ell^{e_j}, e_j = min(j, d_j) (1)(e), the types of
//      ker / coker (phi - c), the Fitting types (1)(b); several T_p read in ONE system of generators (so their commutation is checked).
//  (E) at ell >= 5 (q = 1 only): the same read on the O-piece mod ell^K -- the cokernel presentation M_N / Rel_N (D1's generators of
//      M^S + M^ST, [C, prop:descentN](7)(b)) with H_{N,p} (prop:heilbronn(1), (2), over Z/ell^K: 6 is a unit) -- and the free quotient
//      of thm:descentcert(4): the invariant factors of T_p - c on Lambda (x) Z_(ell) at a non-root c (prop:torsionhecke(4)).
//  (F) the record of thm:descentcert(2), (3): RunDescentHecke.
//
// REFUSALS (one code each, printed "REFUSED [<name>] ..."; the taxonomy of D4's PairRefusal, extended).  An exception that is none of
// these -- an ENGINE_CHECK, a hecke::HeckeInternalError, out_of_range -- is a GENUINE ERROR and propagates (a validation failure):
//   NotPrime, PDividesN        p not prime / p | N (U_p on request only, allow_bad_primes: S26 Z1 / Z2; the default refuses, [R-Q4]);
//   Lift                       the strict lift at p was refused; `lift` carries D4's code (RssCeiling, IntegerCap, DomainCap,
//                              InputTableCertificate, LiftCertificate);
//   DegreeRange                q beyond the built lift / comparison (q_max <= 4 here);
//   RssCeiling                 the predicted RSS of a read above the ceiling, decided before building (rule 19);
//   GroupUncertified           the Smith form of d^{q-1} has r_K != rank_Q (prop:descentmaps(2)(g));
//   ModulePrecision            2 sigma_max > K: the module, (1)(c), (d), (1)(e) beyond ell^{K'} are refused; P_j and the Fitting types
//                              are still reported (a partial refusal: `module_certified` = false);
//   NotEndomorphism            the image of a torsion generator has a non-zero free coordinate, or the relations are not preserved;
//   KerCokerPrecision          ([C] v1.23.0-pre.37 sec:recordschema item 7 (j), the D6 code item) the module is certified but the
//                              types of ker / coker (T_p - c) (prop:torsionhecke(1)(c)) are not certified at the read's ring width
//                              (a rank shortfall of the image lattice, e.g. (1, 200) at ell = 2: 30 < 34 pivots at 2^30) -- a
//                              precision refusal of the family of ModulePrecision, NOT a statement that T_p is no endomorphism;
//   NotCocycle                 d^q F x~_j != 0 mod ell^{K'} (the operator is not a cochain map);
//   IntegerCap / DomainCap     a named cap while the operator is assembled (hecke::HeckeOverflow, std::overflow_error);
//   (not a code) "[HeldOutMismatch]"  NOT a refusal (the D3 audit's m2): a certified coefficient of char_Z disagrees with the held-out
//                              prime -- impossible unless the code, the closed form or the bound is wrong ([C] prop:crtcharpolyN(2), (3));
//                              thrown as hecke::HeckeInternalError whose text starts "[HeldOutMismatch]".  The same for a residue prime
//                              ell' > n + 1 whose shipped presentation has dim H^1 != b_1 ("[ResidueRank]"): a defect, never a refusal.  The
//                              vestigial enumerator D3Refusal::HeldOutMismatch was removed at D5c (C-D8 referee N7): no API returned it.
//
// No settings key, no record key, no golden object, no env variable read here except the read-only HECKE_DESCENT_TREE switch of
// D5 (which never changes a value; D3's 2- and 3-adic Hecke route is the tree under either setting -- the transport is defined on
// every model, and the switch selects the GROUPS' route only).
// ===========================================================================

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "DescentMaps.h"
#include "DescentPair.h"
#include "../Model/IsotropyModelTree.h"

namespace integral_descent {
namespace hecke3 {

using maps::Mat;
using maps::ZMod;
using U32Ctx = engine::ModCtx<uint32_t>;

// ---- refusal codes ------------------------------------------------------------------------------------------------------------
enum class D3Refusal { None, NotPrime, PDividesN, Lift, DegreeRange, RssCeiling, GroupUncertified, ModulePrecision, NotEndomorphism,
                       NotCocycle, IntegerCap, DomainCap, KerCokerPrecision };
const char* D3RefusalName(D3Refusal r);
// IntegerCap / DomainCap for the named caps (via pair::ClassifyPairException), None for everything else (rethrown by the callers).
D3Refusal ClassifyD3Exception(const std::exception& e);

// ---- validation-only fault injection (negative controls through the production route) -----------------------------------------
// One process-wide selector, None by default; set ONLY through ScopedD3Fault by IntegralDescentHeckeValidation's (N) arm, never from
// an env variable, a setting or a CLI token.
enum class D3Fault : uint32_t {
    None = 0,
    DropCoset = 1,          // (B) the last t of every coset block i omitted in T~ (p cosets instead of p + 1)
    BlockSign = 2,          // (B) every even-indexed block term (in assembly order) with its multiplicity + 1, every degree
    SinglePrime = 3,        // (A) the residues at ONE prime only (the bound must refuse to certify the large coefficients)
    NoEisensteinBound = 4,  // (A) the bound without the Eisenstein term, b = r + i ((n+1) bitlen(p) / 2) (the cusp half only)
    PerturbedResidue = 5,   // (A) the residue of c_1 at the first prime + 1 (the held-out prime must catch it)
    ReadBelowModule = 6,    // (D) the read at K = 2 sigma_max - 1 (the module must be refused, P_j and the Fitting types kept)
    PsiDropTerm = 7,        // (C) the first term of Psi_T dropped in every degree (T' is then no cochain map)
    PerturbedDigits = 8,    // (A) the residue of c_r (beyond 2^63) at the middle prime of its prefix + 1 (the held-out prime's digits branch)
};
D3Fault ActiveD3Fault();
struct ScopedD3Fault {
    explicit ScopedD3Fault(D3Fault f);
    ~ScopedD3Fault();
    ScopedD3Fault(const ScopedD3Fault&) = delete;
    ScopedD3Fault& operator=(const ScopedD3Fault&) = delete;
private:
    uint32_t saved_;
};

// =================================================================================================================================
// (A) the characteristic polynomial over Z (prop:crtcharpolyN)
// =================================================================================================================================
// bitlen(x) = floor(log2 x) + 1.
uint32_t BitLen(uint64_t x);
// b_{p,i} = r + i (n + 1) bitlen(p) ((3)); |c_i| <= C(r, i) (1 + p^{n+1})^i <= 2^{b_{p,i}}.
uint64_t CoefficientBitBound(uint32_t r, uint32_t n, uint32_t p, uint32_t i);
// The word mulmod of (4): a * b mod Q for a, b < Q, Q in [2, 2^64) (Q = 0 means 2^64: wrap-around); shift-and-add, never a double
// word.  (a < Q is required; b any word.)
uint64_t WordMulMod(uint64_t a, uint64_t b, uint64_t Q);
// Garner's mixed radix of the residues (each < its prime, primes pairwise distinct odd < 2^32): digits d_k < ell_k with
// c^+ = sum_k d_k prod_{j<k} ell_j in [0, P).  inv[k] = (prod_{j<k} ell_j)^{-1} mod ell_k (GarnerInverses).
std::vector<uint32_t> GarnerInverses(const std::vector<uint32_t>& primes);
std::vector<uint32_t> GarnerDigits(const std::vector<uint32_t>& residues, const std::vector<uint32_t>& primes,
                                   const std::vector<uint32_t>& inv);
// sign test of (4): true iff c^+ > (P - 1) / 2, i.e. c = c^+ - P < 0 (digits compared from the top with (ell_k - 1)/2).
bool MixedRadixNegative(const std::vector<uint32_t>& digits, const std::vector<uint32_t>& primes);
// c mod Q by Horner on the digits (word mulmod); the negative case (c^+ mod Q - P mod Q) mod Q.  Q = 0 means 2^64.
uint64_t MixedRadixModQ(const std::vector<uint32_t>& digits, const std::vector<uint32_t>& primes, bool negative, uint64_t Q);
// |c| < 2^63, decided on the digits: then *value = c (signed).
bool MixedRadixToInt64(const std::vector<uint32_t>& digits, const std::vector<uint32_t>& primes, bool negative, int64_t* value);

struct CrtCoefficient {
    uint32_t i = 0;                 // the coefficient of x^{r - i}
    uint64_t bits = 0;              // b_{p,i}
    bool certified = false;         // sum (bitlen(ell') - 1) over the primes used >= b_{p,i} + 1
    bool fits = false;              // certified and |c_i| < 2^63
    int64_t value = 0;              // fits only
    bool negative = false;          // certified only
    uint32_t primes_used = 0;       // the prefix of the prime list used for this coefficient
    std::vector<uint32_t> digits;   // certified and !fits: the mixed-radix digits (prefix primes)
    std::vector<uint32_t> residues; // NOT certified: the residue vector (one per prime of the list)
    uint64_t mod_2_64 = 0, mod_3_40 = 0;   // certified only: c_i mod 2^64, mod 3^40 (from the digits, word mulmod)
};
struct CrtOptions {
    uint32_t max_primes = 0;        // 0: as many as b_{p,r} + 1 needs (every coefficient certified); else at most this many
    uint32_t threads = 0;           // 0: ENGINE_THREADS or hardware_concurrency, capped at 16
    bool held_out = true;           // one more prime (not used for recovery) checked against every certified coefficient
    uint64_t rss_cap_bytes = 40ull << 30;
};
struct CharpolyZ {
    uint32_t N = 1, n = 0, p = 0, r = 0;
    std::vector<uint32_t> primes;   // the residue primes, descending from 2^30
    std::vector<CrtCoefficient> c;  // i = 0 .. r (c_0 = 1)
    bool determined = false;        // every coefficient certified
    uint32_t held_out_prime = 0;
    bool held_out_ok = true;
    uint32_t b1_shipped = 0;        // dim H^1 of the shipped presentation at the first prime (= r)
    double seconds = 0, residue_seconds = 0, garner_seconds = 0;
    uint64_t rss_pred = 0, rss_meas = 0;
    bool refused = false;
    D3Refusal code = D3Refusal::None;
    std::string refusal;
    // "c_0,c_1,...;fnv=0x<digest>;undetermined=<k>" (integers; a coefficient beyond 2^63 as "[sign,D digits]"; undetermined "?")
    std::string text() const;
    // output rule (5), checkable (the D3 audit's m1): FNV-1a-64 (engine::FnvHasher) over r, then per i = 0 .. r: i; certified: 1 + negative,
    // c_i mod 2^64, c_i mod 3^40; not certified: 0, the number of primes, the residues.  Computed from (N, n, p)'s integers alone, so an
    // independent run (the oracle, a cluster rerun) compares one word.
    uint64_t digest() const;
    uint32_t undetermined() const;  // the coefficients not certified
};
// The residue charpoly at ONE prime ell' >= 5 (ell' > n + 1): high -> low, monic, length r + 1 (r = dim H^1 of the presentation).
std::vector<uint32_t> ShippedResidueCharpoly(uint32_t N, uint32_t n, uint32_t p, uint32_t ellp, uint32_t* h1_out = nullptr,
                                             bool check_rel = true);
// Predicted peak RSS (bytes) of CharpolyOverZ with `threads` concurrent primes.
uint64_t PredictCharpolyRss(uint32_t N, uint32_t n, uint32_t threads, uint32_t p = 0);
CharpolyZ CharpolyOverZ(uint32_t N, uint32_t n, uint32_t p, const CrtOptions& opt = {});
// det(x - A) over F_ell' (A row-major r x r), high -> low, by Hessenberg reduction (O(r^3)).
std::vector<uint32_t> CharpolyFp(std::vector<uint32_t> A, uint32_t r, uint32_t ellp);

// =================================================================================================================================
// (B) the level-N operator on the free model (prop:strictliftN(3))
// =================================================================================================================================
// (slot-26 Z2, UP-Z (b); UPZ_design.md Sec. 2.3, 7.3) how the free-model operator at p | N is held and applied.  All three compute the
// same sum over the ring (bit-identical; HeckeZUpValidation's route arm compares them entrywise): the materialised term table of D3, a
// streamed application (the terms are regenerated per coset block (c, i), never stored; row blocks are disjoint, so the blocks run in
// parallel), and at n = 0 (rho_0 = 1) the prime-independent W x W count matrix of the terms (W = cells(q) m, the kCountOnce pattern of
// HZ-RAT-PERF, built once from a parallel stream and applied by a dense product).  kAuto picks kCountDense at n = 0 when W^2 words fit
// kIsoCountDenseBudget, else kMaterialised within kIsoTermBudget, else kStreamed.  p !| N is always kMaterialised (the shipped route).
enum class IsoRoute : uint8_t { kAuto = 0, kMaterialised = 1, kCountDense = 2, kStreamed = 3 };
const char* IsoRouteName(IsoRoute r);
constexpr uint64_t kIsoCountDenseBudget = 2ull << 30;   // 2 GiB of count matrix (W <= 23170)
constexpr uint64_t kIsoTermBudget = 1ull << 30;         // 1 GiB of materialised terms

class IsoHeckeN {
public:
    // The operator on C^q_I(M_N (x) ctx), q = 0 .. q_top (q_top <= 6 = the lift's top degree).  with_terms_z: also keep the integer
    // term list terms_Z() (S25 EXZ2-P2, audit F9: built only on request, +48 B per term; no value of the reduced route depends on it).
    // (slot-26 Z1, UPZ_design.md Sec. 2.1, 7.1) allow_bad_primes: p | N gives the free-model U_p^{(N)} of [C, prop:strictliftN](9): the level-1 lift PairLift(p) with the index-p transversal of Gamma_0(pN) \ Gamma_0(N).
    // Default false: p | N is refused `PDividesN` as before.
    // (slot-26 Z2) route: only for p | N with allow_bad_primes and without with_terms_z (kMaterialised otherwise); validation may force it.
    IsoHeckeN(uint32_t N, uint32_t n, uint32_t p, const U32Ctx& ctx, uint32_t q_top, bool with_terms_z = false, bool allow_bad_primes = false,
              IsoRoute route = IsoRoute::kAuto);
    IsoRoute route() const { return route_; }
    bool refused() const { return code_ != D3Refusal::None; }
    D3Refusal code() const { return code_; }
    pair::PairRefusal lift_code() const { return lift_code_; }
    const std::string& why() const { return why_; }
    uint32_t N() const { return N_; }
    uint32_t n() const { return n_; }
    uint32_t p() const { return p_; }
    uint32_t D() const { return D_; }
    uint32_t q_top() const { return q_top_; }
    uint64_t modulus() const { return ctx_.p; }  // (S25 EXZ2-P1a) the ring Z/ctx.p the reduced terms live in
    uint32_t cells(uint32_t q) const;            // cells of Iso in degree q
    uint32_t dim(uint32_t q) const { return cells(q) * D_; }
    uint64_t terms(uint32_t q) const { return term_count_.at(q); }   // the number of block terms of T~ on C^q (every route)
    // (S25 EXZ2-P1a, EXPORTZ2_design.md v2 Sec. 2) the integer term list of T~ on C^q, in the order of the reduced terms: block (rb, cb)
    // += mul * rho_n(E) with rb = cell * m + (coset i - 1), cb = cell2 * m + (class a2 - 1), E = v_i s_j^{-1} g gamma w_{a2}^{-1} (det p),
    // mul the lift's integer coefficient (NOT reduced).  For the RNS lanes and the entrywise bound of the free block over Z.
    struct TermZ {
        uint32_t rb, cb;
        hecke::Mat22 E;
        int64_t mul;
    };
    const std::vector<TermZ>& terms_Z(uint32_t q) const;   // ENGINE_CHECKs that the list was requested (with_terms_z)
    bool has_terms_Z() const { return with_terms_z_; }
    // Y = T~ X: X, Y row-major dim(q) x k (Y overwritten).
    // threads: the workers of the streamed and count routes (no value depends on it); the materialised route is serial as before.
    void apply(uint32_t q, const std::vector<uint32_t>& X, uint32_t k, std::vector<uint32_t>& Y, uint32_t threads = 1) const;
    // (D5c) T~_Isob on Isob's own layout: bpos[c] = the position of Iso cell c among the boundary cells of degree q (-1 off Isob); X, Y
    // row-major (#boundary cells * D) x k.  The terms of T~_Iso with a boundary row cell, the rows of rres T~_Iso: with the pair lift's
    // closure (B) this is T~_Isob, rres T~_Iso = T~_Isob rres strictly.  Returns false iff a term with a boundary row cell has a column cell
    // off Isob (the closure fails on this layout: the caller refuses IsobClosure).
    bool apply_boundary(uint32_t q, const std::vector<int32_t>& bpos, const std::vector<uint32_t>& X, uint32_t k,
                        std::vector<uint32_t>& Y) const;
    // The dense matrix (validation only; dim(q)^2 words).
    engine::DenseBuf<uint32_t> dense(uint32_t q) const;
    double build_seconds = 0;

private:
    struct Term {
        uint32_t rb, cb;       // row block (cell c, coset i - 1), column block (cell c2, class a2 - 1): block index c * m + (i - 1)
        uint32_t mul;          // multiplicity reduced into ctx
        hecke::Mat22 E;        // v_i s_j^{-1} g gamma w_{a2}^{-1}, det p
    };
    uint32_t N_ = 1, n_ = 0, p_ = 0, D_ = 0, m_ = 1, d_ = 1, q_top_ = 0;
    U32Ctx ctx_;
    std::vector<std::vector<Term>> terms_;
    std::vector<std::vector<TermZ>> terms_z_;
    bool with_terms_z_ = false;
    IsoRoute route_ = IsoRoute::kMaterialised;
    struct Stream;                                  // the streamed / count routes' data (DescentHeckeOps.cpp)
    std::shared_ptr<Stream> stream_;
    std::vector<uint64_t> term_count_;              // per degree
    void apply_streamed(uint32_t q, const std::vector<uint32_t>& X, uint32_t k, std::vector<uint32_t>& Y, uint32_t threads) const;
    void apply_count(uint32_t q, const std::vector<uint32_t>& X, uint32_t k, std::vector<uint32_t>& Y, uint32_t threads) const;
    std::vector<uint32_t> cells_;
    D3Refusal code_ = D3Refusal::None;
    pair::PairRefusal lift_code_ = pair::PairRefusal::None;
    std::string why_;
};

// =================================================================================================================================
// (C) the ell-local tree model with the transported operator
// =================================================================================================================================
// The comparison of prop:isomodeltree(4) on the code's Iso, built once per process to degree kTreeComparisonTop and certified (D5c
// raised it from 4 to 5 for the tree cusp pair's cochains to degree 5; the maps in degrees <= 4 are unchanged).
constexpr uint32_t kTreeComparisonTop = 5;
const integral_model::IntegralTreeComparison& SharedTreeComparison();

class TreeTransport {
public:
    // ell in {2, 3}; ctx = Z/ell^K (ring2k 2^30 / ring3k 3^19 by default: K = 0); cochains and the operator in degrees 0 .. q_top (<= 5;
    // the tree read itself serves q <= 4, D5c's tree pair q_top = 5).
    TreeTransport(uint32_t ell, uint32_t N, uint32_t n, uint32_t q_top, uint32_t K = 0);
    ~TreeTransport();
    uint32_t ell() const { return ell_; }
    const ZMod& Z() const { return Z_; }
    const U32Ctx& ctx() const { return ctx_; }
    uint32_t q_top() const { return q_top_; }
    uint32_t D() const { return D_; }
    uint32_t rank(uint32_t q) const;             // the cochain rank (J-coordinates) in degree q
    uint32_t full(uint32_t q) const;             // tree cells(q) * D
    uint32_t iso_full(uint32_t q) const;         // Iso cells(q) * D
    // d^q : C^q -> C^{q+1} in J-coordinates (rank(q+1) x rank(q)), q < q_top + 1.
    Mat differential(uint32_t q) const;
    // rank_Q d^q from the Betti recursion (b_0, b_1 of D1's closed form, b_q = 0 for q >= 2).
    uint32_t rank_Q(uint32_t q) const;
    // J-coordinates <-> the full layout (row-major, k columns).
    void to_full(uint32_t q, const std::vector<uint32_t>& U, uint32_t k, std::vector<uint32_t>& V) const;
    void from_full(uint32_t q, const std::vector<uint32_t>& V, uint32_t k, std::vector<uint32_t>& U) const;
    // Phi_T^* : C^q_T (full) -> C^q_I (full);  Psi_T^* : C^q_I (full) -> C^q_T (full).
    void phi_star(uint32_t q, const std::vector<uint32_t>& V, uint32_t k, std::vector<uint32_t>& W) const;
    void psi_star(uint32_t q, const std::vector<uint32_t>& W, uint32_t k, std::vector<uint32_t>& V) const;
    // T'_{T,p} on J-coordinates: U -> from_full psi^* T~ phi^* to_full U.
    void apply_hecke(const IsoHeckeN& T, uint32_t q, const std::vector<uint32_t>& U, uint32_t k, std::vector<uint32_t>& Out,
                     uint32_t threads = 1) const;
    double build_seconds = 0;
    struct Impl;

private:
    uint32_t ell_ = 2, N_ = 1, n_ = 0, q_top_ = 0, D_ = 0;
    ZMod Z_;
    U32Ctx ctx_;
    std::unique_ptr<Impl> impl_;
};

// Predicted peak RSS (bytes) of a tree read of H^q at (ell, N, n): the Smith form of d^{q-1} with its left transforms (A, L, L^{-1}T),
// the generator batch through Iso, the rho memo.
uint64_t PredictTreeReadRss(uint32_t ell, uint32_t N, uint32_t n, uint32_t q);
// Predicted peak RSS (bytes) of the ell >= 5 cokernel read (TorsionOnCokernel), from the closed form before anything is built: the
// generator rows (G x D), their transpose A, SmithLevelsLeft's copy of A and its two D x D transforms (L, L^{-1}T), all 4-byte words;
// G <= d ((m + nu_2)/2 + (m + 2 nu_3)/3) (d generators per <S>- and per <ST>-orbit); plus 256 MiB.
uint64_t PredictCokernelReadRss(uint32_t N, uint32_t n);

// =================================================================================================================================
// (D) the torsion read and the invariants (prop:torsionhecke)
// =================================================================================================================================
using BatchOp = std::function<void(const std::vector<uint32_t>& X, uint32_t k, std::vector<uint32_t>& Y)>;   // c x k row-major
struct TorsionRead {
    uint32_t ell = 0, K = 0, q = 0, c = 0;
    bool ok = false;                       // a read was made (possibly with module_certified = false)
    D3Refusal code = D3Refusal::None;      // None, or the refusal (ModulePrecision is reported with ok = true)
    std::string why;
    uint32_t s = 0, rank_Q = 0;
    std::vector<uint32_t> sigma;           // the torsion exponents sigma_j, j in J (Smith order)
    uint32_t smax = 0, Kp = 0;             // K' = K - sigma_max
    bool module_certified = false;         // 2 sigma_max <= K
    std::vector<Mat> Phi;                  // per operator: |J| x |J|, row i reduced mod ell^{sigma_i}
    // the free quotient (when requested): generators = columns s .. c-1 of L^{-1} (only for a cokernel presentation, no next
    // differential), the matrix of each operator on them mod ell^{K'}
    std::vector<Mat> Free;
    double smith_seconds = 0, apply_seconds = 0;
};
// A Smith form with left transforms by valuation levels (the contract of maps::SmithForm(Z, A, true, false); the pivot search O(rows)
// per column instead of O(rows x cols) per pivot); the read's engine.
maps::Smith SmithLevelsLeft(const ZMod& Z, Mat A);
// d = d^{q-1} (c x a) over Z (reduced into Z); rank_Q its rational rank; dnext = d^q (c' x c) or nullptr; ops the operators F on C^q.
// with_free: also the free block (only meaningful when coker d is H^q itself, i.e. dnext == nullptr).
TorsionRead ReadTorsion(const ZMod& Z, const Mat& d, uint32_t rank_Q, const Mat* dnext, const std::vector<BatchOp>& ops,
                        bool with_free = false);

struct TorsionInvariants {
    bool ok = false;
    std::string why;
    std::vector<uint32_t> a;               // exponents ascending
    std::string canon;                     // the oracle's canonical text (d3_hecke_oracle.py): a=..;P=..;B=..;ker=..;cok=..;fit=..
    std::string layers;                    // P=..;fit=.. only (what needs only a certified form)
    bool full = false;                     // ker / cok for every c in [0, ell^{amax}) computed
    uint32_t kc_k = 0;                     // (ICENSUS-P1) the width 2^k the ker / cok types were read at (provenance, not in canon)
    bool kc_escalated = false;             // the read at k_0 = min(K_max, 2 a_max + 3) fell short and was redone at K_max
    bool kc_precision = false;             // !ok: true = a precision shortfall at the last width tried ([KerCokerPrecision]); false = a
                                           // failed safeguard of the ker / coker read ([NotEndomorphism], a defect, never escalated)
};
// sigma, Phi from a read (Phi rows mod ell^{sigma_i}); module_certified false: only P_j, the Fitting types and the blocks mod
// ell^{min(e_j, K')}; full: ker / coker for every c in [0, ell^{amax}) (else none; bounded by max_full_summands).
// kc_k: 0 = the ladder (k_0 = min(K_max, 2 a_max + 3), then K_max on a precision shortfall, MapRes::precision; any other failure is
// not escalated); else the ker / coker read at this one width only
// (IntegralDescentHeckeValidation's arm (E): a certified type is the same at every width)
TorsionInvariants Invariants(uint32_t ell, const std::vector<uint32_t>& sigma, const Mat& Phi, bool module_certified, uint32_t Kp,
                             bool full = true, uint32_t max_full_summands = 400, uint32_t kc_k = 0);
// Phi_a Phi_b == Phi_b Phi_a on the group (rows mod ell^{sigma_i}).
bool CommuteOnTorsion(uint32_t ell, const std::vector<uint32_t>& sigma, const Mat& A, const Mat& B);
// det(x - B) over Z/ell^e (division-free Berkowitz), high -> low.
std::vector<uint32_t> CharpolyModPrimePower(const Mat& B, uint32_t ell, uint32_t e);

// =================================================================================================================================
// (E), (F) the pieces and the record
// =================================================================================================================================
// One local Hecke datum: the ell-primary torsion of H^q with T_p for the listed p (one system of generators).
struct TorsionEntry {
    uint32_t q = 0, ell = 0;
    std::string route;                     // "tree" (ell = 2, 3), "cokernel" (ell >= 5), "iso" (the free model: oracle)
    std::vector<uint32_t> ps;
    TorsionRead read;
    std::vector<TorsionInvariants> inv;    // per p
    bool commute = true;                   // every pair of T_p commutes on the group (module certified only)
    uint64_t rss_pred = 0;
    double seconds = 0;
    bool refused = false;
    D3Refusal code = D3Refusal::None;
    std::string refusal;
};
// ell in {2, 3}: the tree read with the transported operators (production).
TorsionEntry TorsionOnTree(uint32_t ell, uint32_t N, uint32_t n, uint32_t q, const std::vector<uint32_t>& ps, uint32_t K = 0,
                           uint64_t rss_cap_bytes = 40ull << 30, bool full_invariants = true);
// ell >= 5, q = 1: the cokernel presentation with H_{N,p} over Z/ell^K (production); with_free: the free block too.
// RssCeiling (rule 19, the D3 audit's M1) when PredictCokernelReadRss exceeds rss_cap_bytes, before FixedGeneratorRows.
TorsionEntry TorsionOnCokernel(uint32_t ell, uint32_t N, uint32_t n, const std::vector<uint32_t>& ps, uint32_t K = 0,
                               bool with_free = false, bool full_invariants = true, uint64_t rss_cap_bytes = 40ull << 30);
// ORACLE (validation): the free model C_I itself with T~ directly (no comparison map), dense, any ell (Z/ell^K, ell^K < 2^31).
TorsionEntry TorsionOnFreeIso(uint32_t ell, uint32_t N, uint32_t n, uint32_t q, const std::vector<uint32_t>& ps, uint32_t K);
// ORACLE (validation; D5's arm "the transported T'_p equals the one on Iso^(2)"): the dense Iso^(ell) with T'^(ell) = Psi_ell^* T~ Phi_ell^*.
TorsionEntry TorsionOnIsoEll(uint32_t ell, uint32_t N, uint32_t n, uint32_t q, const std::vector<uint32_t>& ps);

// thm:descentcert(4) at ell >= 5 (no ell-torsion needed): the invariant factors of T_p - c on Lambda (x) Z_(ell) read on the
// cokernel presentation mod ell^k; refused (empty, why) when chi(c) = 0 (k never suffices) or k is too small.
struct FreeQuotient {
    bool ok = false;
    std::string why;
    std::vector<uint32_t> exps;           // the exponents of the invariant factors (ascending; zeros dropped)
};
FreeQuotient FreeQuotientFactors(uint32_t ell, uint32_t N, uint32_t n, uint32_t p, int64_t c, uint32_t K = 0);

struct HeckeRecordOptions {
    uint32_t q_max = 3;                   // torsion in H^1 .. H^{q_max} at 2 and 3 (thm:descentcert(6): q_max = 3 determines all)
    bool charpoly = true, torsion = true;
    bool full_invariants = true;
    CrtOptions crt;
    uint64_t rss_cap_bytes = 40ull << 30;
    std::vector<uint32_t> only_ells;      // non-empty: only these torsion primes (measurement)
};
struct HeckeRecord {
    uint32_t N = 1, n = 0;
    std::vector<uint32_t> ps;
    std::vector<CharpolyZ> chi;           // per p (thm:descentcert(2))
    std::vector<TorsionEntry> tors;       // per (q, ell) (thm:descentcert(3))
    std::vector<uint32_t> bad;            // the torsion primes read (2, 3 and B(n, N))
    double t_chi = 0, t_tors = 0, t_total = 0;
    uint64_t rss_peak = 0;
    // some entry (a char_Z or a torsion read) was refused for a legitimate limit (a cap, a precision limit, DegreeRange, ...): `refusal` is
    // the first one's line and the count.  A defect (HeldOutMismatch, ResidueRank, an ENGINE_CHECK) is never a refusal: it propagates.
    bool refused = false;
    uint32_t refused_entries = 0;
    std::string refusal;
};
// The torsion primes the record reads: {2}, and for n even {3} and B(n, N) (BadPrimesN) when n > 0.
std::vector<uint32_t> RecordTorsionPrimes(uint32_t N, uint32_t n);
HeckeRecord RunDescentHecke(uint32_t N, uint32_t n, const std::vector<uint32_t>& ps, const HeckeRecordOptions& opt = {});

}  // namespace hecke3
}  // namespace integral_descent

#endif
