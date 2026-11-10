// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

#pragma once
#ifndef INTEGRAL_HECKEZ_UTAG_H
#define INTEGRAL_HECKEZ_UTAG_H

// ===========================================================================
// HeckeZUTag -- slot-25 package HZ-UTAG: the two full-algebra tags at level N > 1 (slot-25 plan Sec. 2.2;
// 06-work-packages.md "HZ-UTAG"; 01-math-core.md Sec. 4 "level N > 1, full, negative and upper-bound statements", 02 F3).
//  * chi_Q(U_p), p | N, on H^1(Gamma_0(N); Sym^n Z)/tors by CRT over word primes from the `level_n` machinery (the shipped cokernel
//    route, as HeckeZRational; the eigenvalue bound |a_p| <= p^{n+1} of 01 P 4.2(2), coefficients |c_i| <= binom(b1, i)(1 + p^{n+1})^i).
//  * Integer roots of chi_Q(U_p) (candidates 0, +-1, +-p^j proposed, each CERTIFIED by chi(r) = 0 over Z from residues with the bound
//    |q_j(r)| <= (|r| + 1 + p^{n+1})^{deg}), with their multiplicities (successive quotients).
//  * `non_reduced_certified` (the counting criterion of 02 Sec. 3.3): a reduced full T_Q has d systems, at most d^an of them with
//    U_p = 0 (one per anemic class), so at least d - d^an systems with U_p != 0, each occurring in H^1 at least once
//    ([C, prop:heckeZrat] (2), [GT] `lit:diamond-im-es`); b1 - m_0 < d - d^an (m_0 the multiplicity of 0) refutes it. The count
//    "one U_p = 0 system per anemic class" holds only when N is a power of p (every full system = an anemic class + a U_p-eigenvalue);
//    for any other N the tag is never set, and the text says `nr_scope=N_not_p_power(criterion=0|1)` (audit F1).
//  * `upper_bound_certified` (01 P 5.2(2) with theta = U_p in T^fr): when chi_Q(U_p) splits over Z and its distinct roots number d,
//    (C1) holds for theta = U_p (so T_Q = Q[U_p] is reduced), each Eisenstein system occurs once and each cuspidal one twice
//    ([C, prop:heckeZrat] (2)); then C(f, E) | gcd(g_P^an, a_p(f) - a_p(E)) for an Eisenstein root e and a cusp root a (g_P^an the
//    anemic upper bound of HeckeZRational, [C, prop:heckeZcong]), and C(E, E') | e - e' for two Eisenstein roots.
// Nothing here is reached from a shipped record, report, campaign job id or golden text.
// ===========================================================================

#include "HeckeZRational.h"

#include <cstdint>
#include <string>
#include <vector>

namespace heckez {

struct UTagRoot {
    int64_t r = 0;
    uint32_t mult = 0;
    std::string type;                // "eis" | "cusp" | "" (no attribution)
};
struct UTagBound {
    std::string a, b;                // "eis(1)", "cusp(-8)", ...
    uint64_t bound = 0;              // C(a, b) divides this (1: the components do not meet)
};
struct UTagResult {
    uint32_t N = 0, n = 0, p = 0;
    uint32_t b1 = 0;
    int64_t d = 0, d_an = 0, e_k = 0;
    std::vector<std::string> chi;    // chi_Q(U_p), low to high, canonical hex
    uint32_t m0 = 0;                 // multiplicity of the root 0
    std::vector<UTagRoot> roots;     // certified nonzero integer roots, ascending
    uint32_t split_degree = 0;       // m0 + sum of the multiplicities
    uint32_t distinct = 0;           // distinct roots (0 included when m0 > 0); == d with split => (C1) for theta = U_p
    bool non_reduced_certified = false;
    bool n_is_p_power = true;        // N a power of p: the scope of the non_reduced criterion
    bool non_reduced_criterion = false;   // b1 - m0 < d - d^an (the raw inequality; the tag needs n_is_p_power as well)
    bool upper_bound_certified = false;
    std::vector<UTagBound> bounds;
    std::string gP_an;               // the anemic g_P used ("" if none)
    uint32_t primes_used = 0;
    std::vector<std::string> tags() const;
    std::string text() const;
    uint64_t digest() const;
};

// (slot-25 HZ-RAT-PERF fix round 1, audit F1) U_p on H^1(Gamma_0(N); Sym^n F_ell') at word primes ell', one PrimeRun per prime (index
// order; `threads` workers over the primes, no value depends on them).  Every route returns RunAtPrime(N, n, {p}, ell') entrywise (the
// matrix of U_p in the free-column basis of the sparse presentation is unique; prop:heilbronn (1) is checked on every generator):
//  * kTermTable:     RunAtPrime per prime: the whole term table of eq:heilbronn (|Mer_p| m_N terms) rebuilt at every prime -- the route
//                    before the fix; at (2089, 0) about 10^8 terms, > 12.6 GB, per prime, for about 168 primes;
//  * kCountOnce:     n = 0 and |Mer_p| > m_N (D^2 words <= 512 MiB): rho_0 = (1), so H_{N,p} is the prime-independent integer matrix
//                    counting the admissible pairs (eps, j) with j.eps = i (each count <= |Mer_p| < 2^29 < ell'), built ONCE from slices
//                    of Mer_p on `threads` workers and used at every prime as its own residue;
//  * kDensePerPrime: n > 0 and |Mer_p| > m_N (small D): the D x D matrix per prime by HeilbronnDense (slices, never the whole table);
//  * kAuto:          the first that applies of kCountOnce, kDensePerPrime, else kTermTable.  `used` (optional) reports the route taken.
// UTagStep takes kAuto with s.threads.
enum class UTagRoute : uint8_t { kAuto = 0, kTermTable = 1, kCountOnce = 2, kDensePerPrime = 3 };
// (HZ-RAT-PERF re-audit N1) VALIDATION ONLY: kCountOnce's slices of Mer_p hold about `count_slice_terms` terms (0 = 2^20, the shipped
// size; no value depends on it: the partial count matrices sum exactly in uint32); `count_info` (optional) reports the slices and the
// workers BuildCounts used, so a small cell can exercise the multi-slice, multi-worker sum
struct UTagCountInfo {
    size_t slices = 0;
    uint32_t workers = 0;
};
std::vector<PrimeRun> UTagRuns(uint32_t N, uint32_t n, uint32_t p, const std::vector<uint32_t>& ells, uint32_t threads,
                               UTagRoute route = UTagRoute::kAuto, UTagRoute* used = nullptr, size_t count_slice_terms = 0,
                               UTagCountInfo* count_info = nullptr);
const char* UTagRouteName(UTagRoute r);

// p | N prime, N >= 2, n even. `anemic_P` = the T_p of the anemic g_P (empty: no g_P, Eisenstein--cusp bounds from U_p alone).
UTagResult UTagStep(uint32_t N, uint32_t n, uint32_t p, const std::vector<uint32_t>& anemic_P, const RationalSettings& s = {});

}  // namespace heckez

#endif
