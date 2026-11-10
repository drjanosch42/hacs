// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

#pragma once
#ifndef INTEGRAL_HECKEZ_RATIONAL_H
#define INTEGRAL_HECKEZ_RATIONAL_H

// ===========================================================================
// HeckeZRational -- slot-25 package HZ-RAT: the rational step of the Hecke structure over Z
// (docs/implementation/slot25/00-slot25-plan.md Sec. 2.2; docs/implementation/slot24/heckez/06-work-packages.md "HZ-RAT";
// the mathematics in 01-math-core.md P 1.3, P 2.3, P 4.2, P 5.2/5.3, Thm. 4.1 (C1), (C2), Sec. 4 "Big integers" Q1(a)).
// Contract labels of [C] sec:heckeZ (in [C] since v1.25.0-pre.1, CITE-Z 2026-11-06):
// [C, prop:heckeZrat] = 01 P 1.3, [C, thm:heckeZcert] = 01 Thm. 4.1, [C, prop:heckeZcong] = 01 P 5.3.
//
// WHAT IS COMPUTED, per cell (N, n, algebra), all from residues modulo word primes ell' (CrtPrimes: < 2^30, > n + 1, prime to 6N;
// [C, prop:crtcharpolyN](2): H^1(Gamma_0(N); V (x) F_ell') = L (x) F_ell' with the operators reduced, the shipped cokernel route):
//  * b1 (the F_ell' dimension, equal at every prime used), d = dim M_k(Gamma_0(N)) and d^an from the dimension formulas
//    ([C, prop:heckeZrat] (3), (3')); the check d = (b1 + e_k) / 2;
//  * the count-capped lexicographic theta search and (C1): theta = sum c_p T_p passes iff deg rad(chi_theta mod ell') = d_target at one
//    ell' -- certified, because deg rad(chi mod ell') <= deg f_theta <= dim Q[theta] <= d_target (f_theta the squarefree part over Z);
//  * at the GOOD primes of a primitive theta (deg rad = d_target) rad(chi mod ell') is the reduction of f_theta, so disc f_theta,
//    f_theta itself and, where the Eisenstein system is rational, c_theta(lambda_theta) = f_theta'(lambda_theta) (the Eisenstein--cusp
//    resultant) are read off by residues and recovered by Garner's mixed radix with a certified bound (FMTMP's word route,
//    [C, prop:crtcharpolyN](4), [C, sec:recordschema] item 7 (l)); no big integer enters a certified value;
//  * the factorisation of each such integer: trial division to B = 10^6, valuations by residues (D mod q^j while q^j < 2^64, beyond
//    that the cofactor route X = D q^{-e} with q^e X = D checked beyond the bound), the cofactor by residues (c prod q^e = |D|), then
//    the UNTRUSTED finder (HeckeZFactorFinder.h): below 2^64 nf::factor_u64 + a Pratt certificate verified here (128-bit powmod),
//    above 2^64 a BigInt Miller--Rabin label `probable` (the cofactor stays, `partial(unfactored)`), except that a probable root
//    q < 2^64 of C = q^e (e >= 2) is handed to Pratt and, verified, makes the factorisation complete;
//  * the POOL: every prime above B certified (Pratt) anywhere in the cell is divided out of every value of the cell by residues, in
//    passes repeated to a fixed point (e.g. at (1, 42) the Eisenstein primes 2947939, 1798482437 of g_P split the discriminants' 171-bit cofactor, leaving
//    the 67-bit prime); the pool is a function of the cell, so the result stays schedule-free;
//  * (C2): S = the certified primes of gcd_j D_j over the primitive theta_j (valuations min_j), the cofactor of the gcd (a cofactor
//    above 2^64 is the gcd's only if C | D_j is certified by residues for every j, else it is printed as `upper_bound`);
//  * g_P (the Eisenstein partition, level 1 or e^an_k(N) = 1): gcd_p c_p(lambda_p) over the primitive T_p, factored the same way,
//    with its running values in the order of P.
// Nothing returned by the finder enters a result field unchecked: every factor is re-divided (exact product in 128 bits) and every
// prime above B carries a Pratt certificate verified by VerifyPratt, else it is `probable` and part of the cofactor.
// Caps are COUNTS (RationalSettings::canonical_text), never times. Thread count changes no value (the digest is schedule-free).
// ===========================================================================

#include <cstdint>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

namespace heckez {

// ---- dimension formulas ([C, prop:heckeZrat] (3), (3'); prototype s24/heckez/s01r/dan.py) ----------------------------------------
uint64_t Gamma0Index(uint64_t N);                  // psi(N)
uint64_t Gamma0Cusps(uint64_t N);                  // sum_{t | N} phi(gcd(t, N/t))
int64_t DimSk(uint32_t k, uint64_t N);             // dim S_k(Gamma_0(N)), k even >= 0
int64_t DimMk(uint32_t k, uint64_t N);             // dim M_k(Gamma_0(N))
int64_t DimSkNew(uint32_t k, uint64_t N);          // the new subspace
uint64_t EisAnemicCount(uint32_t k, uint64_t N);   // e^an_k(N)
int64_t DimAnemic(uint32_t k, uint64_t N);         // d^an = sum_{M | N} dim S_k^new(M) + e^an_k(N)

// ---- exact integers from residues (FMTMP's word route) -----------------------------------------------------------------------------
struct ZExact {
    std::vector<uint32_t> primes;    // the residue primes (pairwise distinct, odd, < 2^32)
    std::vector<uint32_t> digits;    // Garner digits of c^+ in [0, P)
    bool negative = false;           // c = c^+ - P
    uint64_t bound_bits = 0;         // the certified input bound |c| < 2^bound_bits
    uint64_t covered_bits = 0;       // sum (bitlen(ell') - 1) >= bound_bits + 1
    std::vector<uint32_t> abs_digits;   // the mixed-radix digits of |c|
    uint32_t significant = 0;        // index past the last nonzero digit of |c|

    bool is_zero() const { return significant == 0; }
    int sign() const { return is_zero() ? 0 : (negative ? -1 : 1); }
    uint64_t abs_bits_upper() const;           // |c| < 2^{this}: sum of bitlen over the significant digits
    uint64_t abs_mod(uint64_t Q) const;        // |c| mod Q, Q >= 2 (Q = 0: 2^64)
    bool abs_u64(uint64_t* v) const;           // |c| < 2^64: *v = |c|
    std::string hex() const;                   // FMTMP's canonical signed hex of c
    std::string abs_hex() const;               // of |c|
};
// (HZ-RAT-PERF) Garner's mixed-radix digits in one O(K^2) Barrett pass, parallel over the later indices (threads >= 1); equal to
// integral_descent::hecke3::GarnerDigits(res, primes, GarnerInverses(primes)) digit for digit.
std::vector<uint32_t> GarnerDigitsFast(const std::vector<uint32_t>& primes, const std::vector<uint32_t>& res, uint32_t threads);
// c from its residues at `primes` (each residue < its prime); requires sum (bitlen - 1) >= bound_bits + 1 (ENGINE_CHECK).
ZExact ZFromResidues(const std::vector<uint32_t>& primes, const std::vector<uint32_t>& residues, uint64_t bound_bits);

// ---- Pratt certificates below 2^64 (certified side) -----------------------------------------------------------------------------
struct PrattCert {
    uint64_t q = 0;                  // the claimed prime
    uint64_t witness = 0;            // a with a^{q-1} = 1, a^{(q-1)/r} != 1 for every prime r | q - 1
    std::vector<std::pair<uint64_t, uint32_t>> factors;   // q - 1 = prod r^e
    std::vector<PrattCert> sub;      // one per factor r > kSmallBound (in the order of `factors`)
};
constexpr uint64_t kSmallBound = 1000000;   // B: trial division, `small` primes (sieve)
bool IsSmallPrime(uint64_t q);              // q <= B and prime (the sieve)
uint64_t PowModU64(uint64_t a, uint64_t e, uint64_t m);   // 128-bit products
// true iff the certificate proves q prime; *nodes counts the certificates visited (cap: node_cap, else false).
bool VerifyPratt(const PrattCert& c, uint32_t node_cap, uint32_t* nodes = nullptr);

// ---- factorisations -----------------------------------------------------------------------------------------------------------
struct PrimePower {
    std::string q_hex;               // the prime (canonical hex)
    uint64_t q = 0;                  // the prime when < 2^64, else 0
    uint32_t bits = 0;
    uint32_t e = 0;
    std::string proof;               // "small" | "pratt" | "probable"
};
struct Factored {
    std::string value_hex;           // the integer (signed)
    uint64_t value_bits = 0;         // |value| < 2^value_bits (from its digits)
    bool zero = false;
    std::vector<PrimePower> factors;     // certified ("small", "pratt") first, ascending; then "probable" (informational)
    std::string cofactor_hex = "0x1";    // |value| / prod (certified q^e)
    uint64_t cofactor_bits = 1;
    std::string cofactor_test = "one";   // "one" | "squarefree_by_size" | "unfactored" | "upper_bound" (a gcd's cofactor that is
                                         // only a multiple of the true one, GcdOf when C | D_j is not certified for every j)
    bool complete() const { return cofactor_test == "one" || cofactor_test == "squarefree_by_size"; }
};

struct RationalSettings {
    uint32_t theta_cap = 64;         // candidates tried in the theta search (units included)
    uint32_t theta_coeff_max = 0;    // 0: binom(d_target, 2)  (01 P 4.2(1))
    uint32_t c1_probe_primes = 3;    // word primes the (C1) degree test is tried at, per candidate
    uint32_t pratt_nodes = 256;      // per prime
    uint32_t pratt_witness_cap = 64; // witnesses a = 2, 3, ... tried per node
    uint32_t mr_rounds = 12;         // the BigInt Miller--Rabin bases (the first mr_rounds primes)
    uint32_t threads = 1;            // word-prime parallelism (no value depends on it)
    // (HZ-RAT-PERF (a)) the hard count cap of the word-prime list (sized from the certified targets, grown on demand); beyond it the
    // step ends in WordPrimeCapRefusal = refused(word_prime_cap).  Provenance: printed in the `word_primes=` tail of canonical_text(),
    // which the record's hashed caps text erases ([C] sec:recordschema item 8 (i)); no certified value depends on it.
    uint32_t word_prime_cap = 1u << 21;
    std::string canonical_text() const;
};

// One factorisation from residues (the certified route plus the untrusted finder). `primes` must cover bound_bits + slack.
// `pool`: certified word primes above B found elsewhere in the cell (each tested by residues, its valuation read by residues; a prime
// of the pool carries the proof it was certified with, `pratt`) -- the pool passes of RationalStep (to a fixed point).
Factored FactorFromResidues(const std::vector<uint32_t>& primes, const std::vector<uint32_t>& residues, uint64_t bound_bits,
                            const RationalSettings& s, const std::vector<uint64_t>& pool = {});
// The production verifier behind the finder (SplitWordCofactor calls it with the finder's claim; the validation's negative controls
// call it with forged claims): each claimed prime above B is Pratt-certified (FindPratt + VerifyPratt), small ones sieve-checked, the
// certified part re-divided out of c; any exponent differing from the claim discards the whole claim (rest = c, nothing certified).
void CertifyWordClaim(uint64_t c, const std::vector<std::pair<uint64_t, unsigned>>& claim, const RationalSettings& s,
                      std::vector<PrimePower>& certified, uint64_t* rest);

// ---- the rational step ---------------------------------------------------------------------------------------------------------
struct ThetaEntry {
    std::vector<uint32_t> coeffs;    // c_p in the order of P
    bool primitive = false;          // (C1): deg f_theta = d_target, certified
    std::string lambda_hex;          // the Eisenstein eigenvalue sum c_p (1 + p^{k-1}) (when eis_rational)
    std::vector<std::string> f;      // f_theta, low to high (hex), primitive only
    Factored disc;                   // disc f_theta, primitive only
    Factored eis_res;                // c_theta(lambda_theta) = f'(lambda), primitive and eis_rational only
    uint32_t primes_used = 0;
};
struct RationalResult {
    uint32_t N = 0, n = 0;
    std::string algebra;             // "full" (N = 1) | "anemic"
    std::vector<uint32_t> P;         // the operator list theta's coefficients are indexed by (RationalStepFull: T_p then U_p)
    uint32_t nU = 0;                 // (slot-26 Z1) the last nU entries of P are U_p, p | N (RationalStepFull); 0 for RationalStep
    uint32_t b1 = 0;
    int64_t e_k = 0, d = 0, d_an = 0, d_target = 0;
    std::string d_source;            // "hES_N" | "hES_N_an"
    bool eis_rational = false;       // e^an_k(N) = 1 (one rational Eisenstein system, T_p -> 1 + p^{k-1})
    std::string c1_status;           // "certified" | "open(cap)"
    std::vector<ThetaEntry> thetas;  // every candidate tried, in search order
    int32_t theta_index = -1;        // the first primitive candidate
    std::vector<uint32_t> discriminants_used;   // indices into thetas
    // (C2)
    struct SEntry { uint64_t ell; uint32_t v_disc; std::string reason; };
    std::vector<SEntry> S_cand;      // the certified primes of gcd_j D_j (reason disc_square / disc_factor)
    Factored disc_gcd;               // the gcd over discriminants_used (factors = min valuations; cofactor of the gcd)
    std::string c2_status;           // "complete" | "partial(unfactored)"
    // g_P (eis_rational only)
    std::vector<uint32_t> gP_ops;    // the primitive T_p used, in P order
    std::vector<std::string> gP_running;   // the running gcd after each, factored text "2^8*691" (+ "*c" when unfactored)
    Factored gP;
    std::string gP_status;           // "upper_bound_certified" | "partial(unfactored)" | "general_case"
    std::vector<std::string> cusp_part;    // c_{theta}(x) = f/(x - lambda) for the chosen theta (hex, low to high), eis_rational only
    std::string settings;            // RationalSettings::canonical_text
    std::string text() const;        // canonical text (fixed order; the digest input)
    uint64_t digest() const;         // FNV-1a-64 of text()
};
// (HZ-RAT-PERF (a)) the word-prime list would exceed RationalSettings::word_prime_cap: the record's word refused(word_prime_cap)
// (a pending refusal: only a later version with a larger cap lifts it -- the cap counts primes, so a faster CRT does not), never an
// ENGINE_CHECK.  Thrown by RationalStep and by HZ-UTAG's UTagStep (whose list obeys the same cap, fix round 1 F7).
struct WordPrimeCapRefusal : std::runtime_error {
    uint32_t cap;
    explicit WordPrimeCapRefusal(uint32_t c)
        : std::runtime_error("refused(word_prime_cap): the rational step needs more than " + std::to_string(c) + " word primes"), cap(c) {}
};
// (HZ-RAT-PERF (b2)) optional inputs that change no value: chi_Z(T_p | H^1/tors) per p in P (b1 + 1 integers, low to high, canonical
// signed hex "0x.."/"-0x.." or decimal), the descent's certified characteristic polynomials.  A unit theta = T_p then takes
// f mod ell' = rad(chi_Z mod ell') instead of the Hecke matrices at ell'; the probe primes compare both routes (ENGINE_CHECK).
struct RationalInputs {
    std::map<uint32_t, std::vector<std::string>> chi_z;
};
// full: N = 1 (d_target = d); anemic: N > 1, P of T_p only (d_target = d^an). Refuses (std::invalid_argument) p | N, n odd, N = 0;
// throws WordPrimeCapRefusal beyond the word-prime cap.
RationalResult RationalStep(uint32_t N, uint32_t n, const std::vector<uint32_t>& P, const RationalSettings& s = {},
                            const RationalInputs* in = nullptr);
// (slot-26 Z1, UP-Z (a); UPZ_design.md Sec. 1.3 R6, 1.4 (d')) the full algebra at N > 1: the same step on the operator list P (T_p,
// p !| N, the anemic record's P) followed by U (U_p for every prime p | N, ascending); theta's coefficients are indexed by that list
// (RationalResult::P, nU = |U|), the (C1) target is d = dim M_k(Gamma_0(N)) and eis_rational = false.  RationalStep's checks on P;
// U must be the primes dividing N, ascending.
RationalResult RationalStepFull(uint32_t N, uint32_t n, const std::vector<uint32_t>& P, const std::vector<uint32_t>& U,
                                const RationalSettings& s = {}, const RationalInputs* in = nullptr);

// Text of a factored integer: "2^8*3*691" (+ "*[c:<hex>]" for an unfactored cofactor, "probable" factors marked "?").
std::string FactoredText(const Factored& f);

// ---- shared helpers for HeckeZUTag -----------------------------------------------------------------------------------------------
// The word primes of the package (CrtPrimes order) excluding divisors of 6N; `count` of them.
std::vector<uint32_t> WordPrimes(uint32_t count, uint32_t N);
// sum (bitlen(ell') - 1) over primes[0 .. k)
uint64_t CoveredBits(const std::vector<uint32_t>& primes, size_t k);
uint32_t BitLen64(uint64_t x);
// Characteristic polynomials (low to high, mod ell') of the operators P at (N, n) on H^1(Gamma_0(N); Sym^n F_ell'), the shipped
// cokernel route; also the dimension.
struct PrimeRun {
    uint32_t ell = 0, h1 = 0;
    std::vector<std::vector<uint64_t>> mats;   // per operator, h1 x h1 row-major
};
PrimeRun RunAtPrime(uint32_t N, uint32_t n, const std::vector<uint32_t>& P, uint32_t ell);
// (HZ-RAT-PERF fix round 1, audit F4) dim H^1(Gamma_0(N); Sym^n F_ell) from the same sparse presentation RunAtPrime reads, without
// any Hecke matrix (= RunAtPrime(...).h1); the chi_Z route checks it against b1 at every word prime it uses.
uint32_t H1DimAtPrime(uint32_t N, uint32_t n, uint32_t ell);
// (HZ-RAT-PERF (b1)) by the Hessenberg route (HeckeZCharpoly.h), O(h1^3); the Berkowitz route (hecke::level::HCharpoly, O(h1^4),
// the route before HZ-RAT-PERF) is kept for the validation's comparison arm only.
std::vector<uint64_t> CharpolyOfCombination(const PrimeRun& r, const std::vector<uint32_t>& coeffs);
std::vector<uint64_t> CharpolyOfCombinationBerkowitz(const PrimeRun& r, const std::vector<uint32_t>& coeffs);
// the word primes [0, count) grown in place (prefix-stable: equal to WordPrimes(count, N))
void ExtendWordPrimes(std::vector<uint32_t>& v, size_t count, uint32_t N);

} // namespace heckez

#endif
