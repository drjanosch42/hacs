// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

#pragma once

// The level-1 q-expansion sum-rule oracle (slot-22 L23-3; census_l23_plan.md v0.3 Sec. 4.3, L23_1_audit.md Sec. 7 item 2;
// C++ port of docs/implementation/slot22/fbin3/sumrules.py, as prototyped modulo ell^K in
// docs/implementation/slot22/l233/qexp_proto.py).
//
// INDEPENDENT of the amalgam, the reader and C_I: S_k (level 1, k = n + 2) by the unitriangular integral basis
// f_j = Delta^j E_4^{3(d - j)} G (j = 1 .. d = dim S_k, G = E_4^a E_6^b with 4a + 6b = k - 12d), echelonised over Z on
// q^1 .. q^d (leading coefficient 1, so every step is exact modulo ell^K); T_p on S_k by a_m(T_p f) = a_{pm}(f) +
// p^{k-1} a_{m/p}(f); the ell-adic valuations of det(X | S_k) as the sums of the Smith valuations of X over Z/ell^K
// (global minimum-valuation pivoting; exact for every invariant < K; K doubles until every invariant is < K).
// Delta = q prod (1 - q^m)^24 (no division by 1728).  Arithmetic: nf::BigInt residues in [0, ell^K).
//
// What the record compares it with (Eichler--Shimura: Lambda^1 (x) Q = S_k^2 + the Eisenstein line, Hecke-equivariantly;
// det is basis-free, so the Smith valuations of an operator on the lattice Lambda^1 sum to v det over Q):
//     sum ed(T_ell | Lambda^1)                  = 2 v det(T_ell | S_k)                       (the Eisenstein line: a unit)
//     sum ed(T_p - (1 + p) | Lambda^1)          = 2 v det(T_p - (1 + p) | S_k) + v(p^n - 1)
//     sum ed_finite(T_p - (1 + p^{n+1}) | ...)  = 2 v det(T_p - (1 + p^{n+1}) | S_k)        (one kernel: the Eisenstein line)
// and at ell = 3 on the iota halves (Lambda^- ~ S_k, Lambda^+ ~ S_k + the Eisenstein line over Q):
//     sum ed(X | Lambda^-) = v det(X | S_k),  sum ed(X | Lambda^+) = v det(X | S_k) + v(lambda_Eis(X)).
// Single-threaded; no setting, no golden object.

#include <cstdint>
#include <map>
#include <string>
#include <vector>

namespace l23 {

struct SumRuleRow {
    uint32_t ell = 0, n = 0, k = 0, dim_s = 0;
    uint32_t K = 0;                         // the working precision that resolved every invariant
    bool resolved = false;                  // every Smith invariant < K (at the last K tried)
    uint32_t vdet_tell = 0;                 // v_ell det(T_ell | S_k)
    uint32_t vdet_tell_minus_1 = 0;         // v_ell det(T_ell - 1 | S_k)  (T_ell^2 - T_ell = T_ell (T_ell - 1))
    std::map<uint32_t, uint32_t> vdet_tp;   // p != ell: v_ell det(T_p - (1 + p) | S_k)
    std::map<uint32_t, uint32_t> vdet_eis;  // p != ell: v_ell det(T_p - (1 + p^{n+1}) | S_k)
    std::map<uint32_t, uint32_t> v_pn1;     // p != ell: v_ell(p^n - 1)
    uint32_t max_invariant = 0;             // the largest Smith valuation met (the precision need)
    uint32_t attempts = 0;
    double seconds = 0;
    std::vector<std::string> log;           // SUMRULE_ORACLE lines, one per attempt
};

// primes: the T_p of the row (T_ell is always computed); k_start = 0 picks max(64, 2 floor((n+1)/(ell+1)) + 64);
// K doubles up to k_cap.  mutate (TEST-ONLY negative control): 1 = drop the p^{k-1} a_{m/p} term of T_p.
SumRuleRow sum_rule_oracle(uint32_t ell, uint32_t n, const std::vector<uint32_t>& primes, uint32_t k_start = 0,
                           uint32_t k_cap = 16384, uint32_t mutate = 0);

// dim S_k at level 1 (k even >= 0; 0 for k odd or k < 12).
uint32_t dim_cusp_forms(uint32_t k);

} // namespace l23
