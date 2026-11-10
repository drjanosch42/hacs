// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

#pragma once
#ifndef HECKE_EXT_LVALUE_BERNOULLI_H
#define HECKE_EXT_LVALUE_BERNOULLI_H

// ===========================================================================
// (hecke-extensions slot 19, WP-LV) EXACT BERNOULLI NUMBERS, IN-HOUSE.
// [E] sec:lvalue, Remark rmk:lvgates ("Bernoulli numbers"): the values B_k / k of
// the Eisenstein / Bernoulli route come by TWO independent routes, which must
// agree (owner's no-CAS rule of 2026-10-03; no external library):
//
//  (X) EXACT.  B_k / k = (-1)^{k/2 - 1} T_{k-1} / (2^k (2^k - 1)) for k >= 2 even,
//      T_{2j-1} the tangent numbers (tan x = sum T_{2j-1} x^{2j-1} / (2j-1)!),
//      which are positive INTEGERS computed by the Brent-Harvey recurrence
//      (additions and multiplications by small integers only).  The fraction is
//      exact but not reduced; every reading of it (the ell-adic valuation, the
//      residue mod ell^m, the comparison with a printed fraction) is exact.
//      The integers are NumberField's nf::BigInt (S24 CL-G1 retired the private
//      unsigned bignum that preceded it here, value-free; CODEAUDIT A6).
//  (V) MOD ell^m.  Voronoi's congruence: for k even, N >= 1 and a coprime to N,
//        (a^k - 1) B_k == k a^{k-1} sum_{j=1}^{N-1} j^{k-1} floor(j a / N)  (mod N).
//      With N = ell^{m + v_ell(k)} and a a primitive root mod ell (so a^k - 1 is a
//      unit when ell - 1 does not divide k), B_k / k mod ell^m follows.  Cost
//      ~ ell^{m + v(k)} modular powers: the route of the deep rungs (k = 342252 at
//      691) where the exact numerator is out of reach.
//
// Both need ell >= 5 prime and (ell - 1) not dividing k (then B_k / k is
// ell-integral: von Staudt-Clausen, and Kummer for the k-part).  Pure host
// arithmetic; no pipeline object; writes nothing.
// ===========================================================================

#include "../../NumberField/BigInt.h"

#include <cstdint>
#include <string>
#include <vector>

namespace hecke_ext {
namespace lvalue {

// x mod q for x >= 0 (q >= 1), by the base-2^32 magnitude.
uint64_t ModU64(const nf::BigInt& x, uint64_t q);

// T_1, T_3, ..., T_{2N-1} (index j-1 holds T_{2j-1}).
std::vector<nf::BigInt> TangentNumbers(uint32_t N);

// B_k / k = sign * num / den, exact, NOT reduced (num = T_{k-1}, den = 2^k (2^k - 1)).
struct BernoulliOverK {
    uint32_t k = 0;
    int sign = 1;          // (-1)^{k/2 - 1}
    nf::BigInt num, den;
};
// k >= 2 even (checked).
BernoulliOverK ExactBernoulliOverK(uint32_t k);

// v_ell of a non-zero x >= 0; the unit part is returned mod q (q a power of ell, may be 1).
uint32_t ValuationAndUnitPart(const nf::BigInt& x, uint64_t ell, uint64_t q, uint64_t& unit_mod_q);

// ell-adic reading of B_k / k: v = v_ell(B_k / k) (>= 0 required), and the
// residue mod ell^m (0 <= residue < ell^m).  ell >= 5 prime, (ell - 1) does not divide k,
// ell^m < 2^62.  `exact` selects route (X); otherwise route (V).
struct BernoulliReading {
    uint32_t k = 0;
    uint64_t ell = 0;
    uint32_t m = 0;
    uint32_t v = 0;               // route (X): the exact valuation; route (V): min(v, m) (v >= m reads as m)
    bool v_exact = false;         // route (X), or route (V) with residue != 0
    uint64_t residue = 0;         // B_k / k mod ell^m
};
BernoulliReading BernoulliModEllPowerExact(uint32_t k, uint64_t ell, uint32_t m);
BernoulliReading BernoulliModEllPowerVoronoi(uint64_t k, uint64_t ell, uint32_t m);

// The exact rational B_k (not B_k / k) as (sign, reduced numerator, reduced denominator)
// compared with a printed fraction P / Q (Q > 0): true iff B_k == P / Q exactly.
bool BernoulliEquals(uint32_t k, bool negative, const std::string& P_decimal, const std::string& Q_decimal);

// ---- (slot 19, WP-LV2) TWISTED BERNOULLI NUMBERS B_{k, omega^j} ------------------------------
// [E] cl:lvtwist, sec:lvnorm row (E_chi): L(1-k, chi) = -B_{k,chi}/k, chi = omega^j (omega the Teichmuller
// character mod ell, conductor ell), and [U] lem:u-twistvalues.  B_{k,chi} lies in Q(mu_{ell-1}); it is read
// in Z_ell through omega (the embedding fixed by omega(a) == a mod ell), modulo ell^m.  Two routes, which
// must agree:
//  (XT) THE DEFINITION.  B_{k,chi} = ell^{k-1} sum_{a=1}^{ell} chi(a) B_k(a/ell)
//       = sum_i C(k,i) B_i ell^{i-1} S_{k-i},  S_r = sum_{a<ell} omega(a)^j a^r,
//       with the EXACT B_i of route (X) (B_1 = -1/2); omega(a) = a^{ell^{P-1}} mod ell^P.  Terms with
//       i - 1 >= P vanish mod ell^P (v(B_i) >= -1, von Staudt-Clausen), so only B_0 .. B_P enter.
//  (KV) KUMMER.  B_{k,omega^j}/k = -L_ell(1-k, omega^{k+j}) and s |-> L_ell(1-s, omega^a) = G_a(u^s - 1),
//       G_a in Z_ell[[T]] (a != 0 mod ell-1), so for k' == k + j (mod ell - 1), k' == k (mod ell^{m-1}):
//       B_{k,omega^j}/k == (1 - ell^{k'-1}) B_{k'}/k' (mod ell^m), the right side by route (V).
// Both need ell >= 5 prime, j != 0 mod (ell - 1), j == k mod 2 (else B_{k,chi} = 0).  (XT) reads v exactly
// up to its working precision (v_exact); (KV) only below m.  `k_prime` (KV) is the k' used.
BernoulliReading TwistedBernoulliOverK(uint32_t k, uint64_t ell, uint32_t j, uint32_t m);
BernoulliReading TwistedBernoulliKummer(uint32_t k, uint64_t ell, uint32_t j, uint32_t m, uint64_t* k_prime = nullptr);
// omega(a)^j mod ell^P (the Teichmuller lift), a a unit mod ell.
uint64_t TeichmullerPowMod(uint64_t a, uint32_t j, uint64_t ell, uint32_t P);

// helpers (exposed for the validation)
uint64_t PowModU64(uint64_t b, uint64_t e, uint64_t q);
uint64_t InvModU64(uint64_t a, uint64_t q);          // a a unit mod q (gcd 1); throws otherwise
uint64_t PrimitiveRootModPrime(uint64_t ell);
uint32_t ValuationU64(uint64_t x, uint64_t ell);     // x > 0

} // namespace lvalue
} // namespace hecke_ext

#endif // HECKE_EXT_LVALUE_BERNOULLI_H
