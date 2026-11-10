// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

#pragma once
#ifndef INTEGRAL_HECKE_MAZUR_EIS_H
#define INTEGRAL_HECKE_MAZUR_EIS_H

// ===========================================================================
// HeckeMazurEis -- slot-25 package H3: Mazur's Eisenstein ideal at prime level, integrally (a validation against known theorems).
// docs/implementation/slot24/heckez/06-work-packages.md "H3"; 05-hida-targets.md Sec. 3 (with the A7-M1 note of Sec. 3.5);
// docs/implementation/slot25/00-slot25-plan.md wave 2. Contract labels of [C] sec:heckeZ (in [C] since v1.25.0-pre.1, CITE-Z 2026-11-06):
// [C, prop:heckeZeiscert] is the Eisenstein certificate with its valuation check (A7-M1).
//
// SETTING. N prime, weight 2 (n = 0), ell >= 5 prime with ell | nu := num((N - 1)/12) (Mazur, IHES 47, II.9.7: T/I = Z/nu).
// On H^1(Gamma_0(N); Z)/tors (rank b1 = 2g + 1) T_q (q != N) has chi_Z(T_q) = (x - 1 - q) c_q(x)^2, c_q of degree g the
// characteristic polynomial of T_q on S_2(Gamma_0(N)) (Eichler--Shimura; the monic square root is unique).
//
// WHAT IS COMPUTED, per cell (N, ell):
//  * c_q over Z, from residues: at each word prime ell' (HZ-RAT's WordPrimes / RunAtPrime, the shipped cokernel route) chi(T_q) mod
//    ell' is divided by (x - 1 - q) (remainder checked 0) and its unique monic square root h is taken top-down (2 a unit) and checked
//    (h^2 = chi / (x - 1 - q) mod ell', counted in `structure_failures`); Garner (ZFromResidues) recovers c_q with the bound
//    |coeff_i| <= binom(g, i) (2 sqrt q)^i (Ramanujan--Petersson for weight 2, Eichler--Shimura / Weil: STD), so c_q mod ell^K is exact;
//  * the Hensel factor f_{q,m} of c_q at 1 + q over Z_ell, mod ell^K (ell^K < 2^62): in y = x - 1 - q, C(y) = c_q(y + 1 + q) =
//    y^e H(y) mod ell with H(0) a unit; linear Hensel lifting of (y^e, H) to C = F * H~ mod ell^K, re-checked (F monic, F = y^e mod ell,
//    F * H~ = C mod ell^K) -- the unique such factor (Hensel), so certified mod ell^K;
//  * the good-prime finder (UNTRUSTED as to its choice, CHECKED as to its output): candidates q = Mazur good primes (II.16, p. 124;
//    hecke::level::MazurGoodPrime) in increasing order; a candidate is ACCEPTED iff the valuation check v_ell(F(0)) = v_ell(F_q(1 + q))
//    = v_ell(nu) holds (A7-M1: goodness alone does not keep another residue block out of F_q, counterexample (61, 5, 37)); the accepted
//    q is recorded and its goodness re-checked by two independent modular powers (q !== 1 mod ell, q^{(N-1)/ell} != 1 mod N);
//  * the checks of 05 Sec. 3.1 / 06 "H3":
//    (a) deg F = e_W := (dim W_{N,ell} - 1)/2 of [C, LN6]'s reading (hecke::level::ReadEisensteinR3Scan on {2, 3, 5, 7} \ {N}), and
//        (e >= 2) <=> Merel's criterion [C, LN5] (hecke::level::MerelCriterionLN5);
//    (b) when ell || nu: F(y) is Eisenstein (every lower coefficient divisible by ell, v(F(0)) = 1);
//    (c) v_ell(F(0)) = v_ell(nu) for the accepted q (true by acceptance; the count is of cells where some good q passes);
//    (d) reported at ell^2 | nu: the Newton polygon of F, the embedding dimension of Z_ell[x]/(F) (1 iff DVR), irreducibility over
//        Q_ell when the polygon decides it.
// Nothing here enters a record, golden object or campaign job id. Thread count changes no value.
// ===========================================================================

#include <cstdint>
#include <string>
#include <vector>

namespace heckez {

struct MazurEisSettings {
    uint32_t batch = 4;              // good primes tried per word-prime pass
    uint32_t max_candidates = 16;    // good primes tried in all (cap: a count)
    uint32_t q_max = 2000;           // candidates q < q_max
    uint32_t threads = 1;            // word-prime parallelism (no value depends on it)
    std::vector<uint32_t> forced_q;  // non-empty: exactly these q, good or not (the negative control); the finder is off
    std::string canonical_text() const;
};

struct MazurEisCandidate {
    uint32_t q = 0;
    bool good = false;               // MazurGoodPrime
    uint32_t e = 0;                  // deg F_q (the multiplicity of 1 + q in c_q mod ell)
    uint32_t v_f = 0;                // v_ell(F_q(1 + q)) (K if 0 mod ell^K)
    bool accepted = false;           // good and v_f == v_ell(nu)
};

struct MazurEisResult {
    uint32_t N = 0, ell = 0;
    uint64_t nu = 0;                 // num((N - 1)/12)
    uint32_t v_nu = 0;               // v_ell(nu)
    uint32_t g = 0, b1 = 0;
    uint32_t K = 0;                  // the ell-adic precision (ell^K < 2^62)
    uint32_t primes_used = 0;        // word primes (summed over passes)
    uint32_t structure_failures = 0; // word primes where chi(T_q) != (x - 1 - q) h^2
    std::vector<MazurEisCandidate> candidates;   // in the order tried
    int32_t accepted_index = -1;
    uint32_t q = 0;                  // the accepted q (0: none)
    bool first_good_rejected = false;   // the first good q failed the valuation check (A7-M1 phenomenon)
    std::vector<uint64_t> F;         // F(y) = f_{q,m}(y + 1 + q) mod ell^K, low to high, monic of degree e
    std::vector<std::string> c_hex;  // c_q over Z for the accepted q (canonical signed hex, low to high)
    uint32_t e = 0;                  // deg F
    // (a)
    uint32_t dimW = 0, loewy = 0, e_W = 0;
    std::vector<uint32_t> ln6_qs;
    bool merel = false;
    // (d)
    std::vector<uint32_t> newton_v; // v_ell(F_i), i = 0 .. e (K for 0 mod ell^K)
    std::string newton;              // the lower convex hull as "(0,v0)-(i1,v1)-...-(e,0)"
    uint32_t embdim = 0;             // dim m/m^2 of Z_ell[y]/(F): 1 iff DVR (e = 1, or v(F(0)) = 1)
    std::string irreducible;         // "yes" | "no" | "undecided" (from the Newton polygon over Q_ell)
    // the checks
    bool a_ok = false, b_applicable = false, b_ok = false, c_ok = false, good_q_ok = false;
    std::string settings;
    std::string text() const;        // canonical (fixed order; the digest input)
    uint64_t digest() const;         // FNV-1a-64 of text()
};

// N prime >= 11, ell >= 5 prime, ell | num((N - 1)/12) (std::invalid_argument otherwise).
MazurEisResult MazurEisStep(uint32_t N, uint32_t ell, const MazurEisSettings& s = {});

// The H3 list: C4 (04 Sec. 3.2: every prime N < 200 and every prime N < 900 with some ell^2 | nu, each with every ell >= 5 | nu) united with
// [C, LN6]'s cells (N <= 2000 prime, ell in {5, 7, 11, 13}, ell | nu), sorted by (N, ell), duplicates once.
struct MazurEisCell { uint32_t N, ell; bool c4, ln6; };
std::vector<MazurEisCell> MazurEisCellList();

}  // namespace heckez

#endif
