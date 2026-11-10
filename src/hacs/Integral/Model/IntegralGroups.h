// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

#pragma once
#ifndef INTEGRAL_MODEL_INTEGRAL_GROUPS_H
#define INTEGRAL_MODEL_INTEGRAL_GROUPS_H

// ===========================================================================
// Line 2, M0 (hecke-extensions slot 9, WP-11): the integral groups
// H^q(Gamma; Sym^n Z^2), q <= q_max, read off the isotropy-resolution model
// C^._I(Z) of [C, def:isomodel] (Integral/Model/IsotropyModel, WP-10) by the
// certified-exponent rule [C, prop:certifiede].  [C] =
// docs/latex/gl2_contract.tex (v1.8.1), cited by label.
//
// WHAT IS COMPUTED, AND FROM WHAT (no bignum, no integer beyond a word):
//
//   * rank_Q d^q for q = 0 .. q_max, as the rank of d^q over ONE good prime
//     ell' (ell' >= 5, ell' > n + 1): [C, prop:certifiede](4) for C = C_I(Z).
//     Then rank H^q = w_q (n+1) - rank_Q d^q - rank_Q d^{q-1}
//     ([C, prop:certifiede](2)).
//   * at each requested prime ell (2 and 3 included -- [C,
//     prop:certifiede](5): the rule applies to sigma^{<= q_max+1} C_I(Z) at
//     EVERY ell), and for each differential d^q SEPARATELY ("K certified per
//     degree"), the first exponent K >= 2 with r_K(d^q) = rank_Q d^q, where
//     r_K(d^q) is the number of invariant factors of colspan(d^q mod ell^K)
//     over Z/ell^K (colspan ~= (+)_t Z/ell^{K - a_t}, one summand per Smith
//     entry ell^{a_t} with a_t < K).  At that K every a_t < K
//     ([C, prop:certifiede](1)) and the non-unit elementary divisors of d^q
//     over Z_(ell) are READ OFF exactly: tors H^{q+1}(C)_(ell) =
//     (+) Z/ell^{a_t}, a_t >= 1 ([C, prop:certifiede](2)).
//   * the group type Z^{r_q} (+) (+)_ell tors_ell.  The primes at which
//     H^.(Gamma; Sym^n Z^2) can have torsion are those of
//     [C, prop:badprimes]; the CALLER chooses which primes to certify and a
//     prime not certified contributes nothing (the caller's responsibility,
//     printed in every summary).
//
// The model complex C_eq(Z/ell^e) of the SHIPPED 7-vertex pipeline at
// ell in {2, 3} is NOT touched here: it is the Bredon object, not the
// reduction of an integral complex, and the rule does not apply to it
// ([C, prop:certifiede](5), [C, prop:bredon](3)).  Everything below is on
// C_I only.  No settings key, no record key, no golden object.
//
// DEPENDENCIES: IsotropyModel.h and ModArithmetic/ only (the WP-10 rule for
// Integral/Model/): nothing from HeckeExt/, nothing from the 7-vertex pipeline.
// ===========================================================================

#include <cstdint>
#include <map>
#include <string>
#include <vector>

#include "IsotropyModel.h"

namespace integral_model {

// A finitely generated abelian group Z^free_rank (+) (+)_ell (+)_i Z/ell^{a_i}.
// torsion[ell] lists the exponents a_i >= 1, DESCENDING; no empty entries.
struct AbelianGroupType {
    uint32_t free_rank = 0;
    std::map<uint32_t, std::vector<uint32_t>> torsion;

    bool operator==(const AbelianGroupType& o) const { return free_rank == o.free_rank && torsion == o.torsion; }
    bool operator!=(const AbelianGroupType& o) const { return !(*this == o); }
    bool is_zero() const { return free_rank == 0 && torsion.empty(); }
    // Adds Z/ell^a (a >= 1), keeping the list descending.
    void add_torsion(uint32_t ell, uint32_t a);
    // The ell-primary part only (free rank 0).
    AbelianGroupType primary_part(uint32_t ell) const;
    // Primary form, e.g. "Z^3 + Z/2 + Z/4 + Z/3 + Z/13" (primes ascending,
    // exponents ascending within a prime -- the probe's order); "0" if zero.
    std::string to_string() const;
    // Invariant-factor form d_1 | d_2 | ... (each > 1, ascending), e.g.
    // Z/4 + Z/3 -> {12}; ENGINE_CHECKs every d_i < 2^63.
    std::vector<uint64_t> invariant_factors() const;
    // "Z + Z/12" style text from invariant_factors().
    std::string invariant_factor_string() const;
    // The composition length of G (x) Z/ell^f: f * free + sum min(a, f)
    // (over ell's exponents only).
    uint32_t tensor_length(uint32_t ell, uint32_t f) const;
    // The composition length of Tor(G, Z/ell^f): sum min(a, f).
    uint32_t tor_length(uint32_t ell, uint32_t f) const;
};

// The certified Smith data of one differential d^q of C_I(Z) at one prime.
struct CertifiedDifferential {
    uint32_t q = 0;
    uint32_t ell = 0;
    uint32_t rank_Q = 0;                 // from the good prime
    uint32_t K = 0;                      // the first K >= 2 with r_K == rank_Q (0 if none within budget)
    bool certified = false;
    std::vector<uint32_t> r_K_tried;     // r_K(d^q) at K = 2, 3, ..., in order
    std::vector<uint32_t> nonunit;       // a_t >= 1 of the Smith entries ell^{a_t}, descending (certified only)
};

// One prime's certification of d^0 .. d^{q_max}.
struct PrimeCertification {
    uint32_t ell = 0;
    uint32_t K_budget = 0;
    std::vector<CertifiedDifferential> d;   // d[q], q = 0 .. q_max
    bool all_certified() const;
    uint32_t K_max() const;                 // max over d[q].K
};

// H^q(Gamma; Sym^n Z^2), q = 0 .. q_max, from C_I(Z) alone.
struct ModelIntegralCohomology {
    uint32_t n = 0;
    uint32_t q_max = 0;
    uint32_t good_prime = 0;
    std::vector<uint32_t> rank_Q;           // rank_Q d^q, q = 0 .. q_max
    std::vector<uint32_t> cochain_rank;     // w_q (n + 1), q = 0 .. q_max + 1
    std::vector<AbelianGroupType> H;        // q = 0 .. q_max
    AbelianGroupType tors_H_top;            // tors H^{q_max+1}: the non-unit divisors of d^{q_max}
    std::vector<PrimeCertification> primes; // in the order requested
    bool all_certified() const;
};

// rank of d^q over F_{good_prime}, q = 0 .. q_max.  ENGINE_CHECKs
// good_prime >= 5 and good_prime > n + 1 ([C, prop:certifiede](4)).
std::vector<uint32_t> RationalRanksAtGoodPrime(const IsotropyModel& M, uint32_t n, uint32_t good_prime);

// K budget for a prime: the largest K with ell^K < 2^31 (uint32_t contexts).
uint32_t DefaultKBudget(uint32_t ell);

// Certifies d^0 .. d^{q_max} of C_I(Z) at ell, one K per differential,
// K = 2, 3, ... up to K_budget.  rank_Q from RationalRanksAtGoodPrime.
// ENGINE_CHECKs r_K <= rank_Q at every K ([C, prop:certifiede](1)): a larger
// r_K would mean the "good" prime was not good.
PrimeCertification CertifyAtPrime(const IsotropyModel& M, uint32_t n, uint32_t ell,
                                  const std::vector<uint32_t>& rank_Q, uint32_t K_budget = 0);

// The assembly on C_I alone: ranks at `good_prime`, torsion at every prime of
// `primes` (each certified per degree).  An uncertified differential leaves
// its prime's torsion in that degree unread; all_certified() says so.
ModelIntegralCohomology AssembleFromModel(const IsotropyModel& M, uint32_t n, const std::vector<uint32_t>& primes,
                                          uint32_t good_prime);

// The universal-coefficient prediction of [C, prop:certifiede](3): the
// composition length of H^q(C_I (x) Z/ell^f) for q = 0 .. q_max, from the
// certified groups: length(H^q (x) Z/ell^f) + length(Tor(H^{q+1}, Z/ell^f)).
std::vector<uint32_t> PredictedLengths(const ModelIntegralCohomology& G, uint32_t ell, uint32_t f);

// The primes ell >= 5 of [C, prop:badprimes](1) for n > 0 even ("Dickson
// primes"): 5 <= ell <= n - 1 with #{(a, b) >= 0 : a(ell+1) + b(ell^2-ell) = n} > 0;
// with that count (the number of cyclic ell-primary summands of H^1).
std::vector<std::pair<uint32_t, uint32_t>> DicksonPrimes(uint32_t n);

}  // namespace integral_model

#endif
