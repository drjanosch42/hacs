// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

#pragma once
#ifndef INTEGRAL_HECKEZ_EISUP_H
#define INTEGRAL_HECKEZ_EISUP_H

// ===========================================================================
// HeckeZEisUp -- slot-26 package HZ-EISUP: the Eisenstein U_p systems of the full algebra at level N > 1
// (docs/implementation/slot26/HZEISUP_design.md v0.2, its audit HZEISUP_design_audit.md; [C] 1.27.0-pre.1 prop:heckeZeisup and item 8 (d')(ii)
// "The components and the maximal ideals from HZ-EISUP on").
//
// WHAT IS COMPUTED, for a `full` record at N > 1 with (C1) certified and e^an_k(N) = 1 (InScope):
//  * q0 = the first operator of the list (G1.P[0], a T_p with p !| N), c0 = 1 + q0^{k-1}, E = ker(T_q0 - c0) on L (x) Q, the Eisenstein part
//    (prop:heckeZeisup (1)); at word primes q (the full step's retained runs first, then further primes of the SAME list by RunAtPrime)
//    K_q = ker(T_q0 - c0) on L / q, skipped unless dim K_q = e_k (C-E1; (2));
//  * on K_q the restrictions of every operator; by CRT under the proved bounds of (1), (3): e_theta = charpoly(theta | E), g_p =
//    charpoly(U_p | E) for every p | N, charpoly(T_p | E) for every T_p of the list (C-E3);
//  * the integer roots of every g_p from C_p = {0, +-1, +-p^j : 1 <= j <= k - 1}, certified by exact division (C-E4; the route requires
//    g_p to split over C_p for every p | N, (4));
//  * the identification at one prime q* (the first used prime with e_theta mod q* separable and the distinct roots of every g_p pairwise
//    incongruent mod q*): the theta-eigenlines of K_q*, on each the scalar of every U_p, the unique root of g_p in its residue class (4);
//  * the exact closure (C-E5): lambda_j(theta) = sum_{T_p} c_p (1 + p^{k-1}) + sum_{p | N} c_{U_p} lambda_j(U_p) distinct, prod_j (x -
//    lambda_j(theta)) = e_theta, lambda_j(theta) = the theta-eigenvalue of line j mod q*; g_p = prod_j (x - lambda_j(U_p)); C-E2: e_theta
//    divides f_theta exactly, c_theta = f_theta / e_theta of degree d - e_k.
// Every failure of C-E2 .. C-E5 contradicts a proved statement or exposes a defect: ENGINE_CHECK (no record is written).  The word-prime cap
// (RationalSettings::word_prime_cap) is the only refusal (WordPrimeCapRefusal, the whole record's refused(word_prime_cap)).
// The closed form (prop:heckeZeisup (5), PredictedEisUp) is a SECOND ROUTE only: no record field depends on it.
//
// DETERMINISM.  Fixed prime order (CrtPrimes without the divisors of N, the full step's own list), deterministic skip, CRT in index order,
// exact integers (nf::BigInt for the identities over Z; certified values from residues under proved bounds).  No value -- and no byte of
// text() -- depends on the thread count, on reuse vs. recompute of the runs, or on the ISA route of RunAtPrime.  `reused` / `computed`
// are provenance (G10), outside text().
// ===========================================================================

#include <cstdint>
#include <string>
#include <vector>

#include "HeckeZRational.h"

namespace heckez {
namespace eisup {

// (design D3, audit m1) the scope of the route: e^an_k(N) = 1 written out (RationalResult::eis_rational is false on every full record).
bool InScope(uint32_t N, uint32_t n);

// VALIDATION ONLY: tampered inputs (HeckeZEisUpValidation arm (f)); each must be caught.  Default: nothing tampered.
struct EisUpTamper {
    int32_t ek_delta = 0;            // e_k + delta: C-E1 is never satisfied (the cap refusal on a small cap)
    bool f_plus_one = false;         // f_theta's constant coefficient + 1 (C-E2)
    bool theta_after = false;        // theta's first coefficient + 1 after e_theta (C-E5)
    bool drop_top_candidate = false; // C_p without p^{k-1} (C-E4)
    // swap lambda(U_<swap_p>) between the lines whose tuples are swap_a and swap_b (decimal, p | N ascending) after the identification
    uint32_t swap_p = 0;
    std::vector<std::string> swap_a, swap_b;
};

struct EisUpInput {
    uint32_t N = 0, n = 0;
    std::vector<uint32_t> ops;       // the operator list (RationalResult::P of RationalStepFull): T_p ascending, then U_p (p | N) ascending
    uint32_t nU = 0;                 // the last nU entries of ops are U_p
    std::vector<uint32_t> theta;     // theta's coefficients on ops ((C1) certified)
    std::vector<std::string> f;      // f_theta (exact, canonical signed hex, low to high, monic of degree d)
    int64_t e_k = 0, d = 0;
    uint32_t b1 = 0;
    RationalSettings rs;             // threads (fallback runs), word_prime_cap
    const std::vector<PrimeRun>* runs = nullptr;   // the full step's retained runs (RationalStepFull's `keep`), may be null
    bool force_recompute = false;    // ignore `runs` (the bit-identity arm)
    const EisUpTamper* tamper = nullptr;
};

struct EisLine {
    std::vector<std::string> U;      // lambda_j(U_p), p | N ascending (canonical signed hex)
    std::string lambda_theta;        // lambda_j(theta) (canonical signed hex)
};

struct EisUpResult {
    uint32_t N = 0, n = 0, nU = 0;
    std::vector<uint32_t> ops, theta;
    int64_t e_k = 0, d = 0;
    uint32_t q0 = 0;
    std::vector<std::string> e_theta;                  // charpoly(theta | E), low to high
    std::vector<std::vector<std::string>> g;           // per p | N ascending: charpoly(U_p | E)
    std::vector<EisLine> lines;                        // ordered by the tuple (lambda_j(U_p))_{p | N asc}, signed, lexicographic ascending
    std::vector<std::string> c_theta;                  // f_theta / e_theta, low to high (degree d - e_k)
    uint32_t primes_used = 0;                          // word primes whose residues were read (dim K_q = e_k)
    std::vector<uint32_t> skipped;                     // word primes with dim K_q != e_k (C-E1), in list order
    uint32_t id_prime = 0;                             // q*
    // provenance (not in text())
    uint32_t reused = 0;                               // retained runs of the full step used
    uint32_t computed = 0;                             // runs computed here (the fallback)
    double secs = 0;

    std::vector<uint32_t> Uprimes() const;             // the p | N ascending (= the last nU entries of ops)
    std::string text() const;                          // canonical text (fixed order; the digest input)
    uint64_t digest() const;                           // FNV-1a-64 of text()
};

// The route.  Requires InScope(N, n), nU >= 1, theta / f / e_k / d / b1 consistent (ENGINE_CHECK); throws WordPrimeCapRefusal beyond the cap.
EisUpResult EisUpStep(const EisUpInput& in);

// (prop:heckeZeisup (5), the second route) the tuples (lambda(U_p))_{p | N asc} for N squarefree: prod_{p | N} {1, p^{k-1}} (k >= 4), without
// (p)_{p | N} (k = 2), each as canonical signed hex, sorted like EisUpResult::lines.  *predicted = false (and empty) for N not squarefree.
std::vector<std::vector<std::string>> PredictedEisUp(uint32_t N, uint32_t k, bool* predicted);

// ---- the attribution at ell (design Sec. 1.5; C-E7, audit M4) -------------------------------------------------------------------------
// lambda_j(theta') for theta' = sum c_i X_i over an operator list `thP` (an entry p with N % p != 0 is T_p, acting on E by 1 + p^{k-1}
// ((F2), T_ell included); an entry p | N is U_p, acting on line j by lines[j].U), exact (canonical signed hex).
std::string LambdaOf(const EisUpResult& R, size_t j, const std::vector<uint32_t>& thP, const std::vector<uint32_t>& coeffs);
// The canonical hex of an integer mod ell, in [0, ell).
uint64_t HexModEll(const std::string& hex, uint64_t ell);
// hex + delta (canonical signed hex; the validation's tampers)
std::string HexPlus(const std::string& hex, int64_t delta);
struct Attribution {
    std::vector<uint64_t> cbar;      // c-bar_{theta'} = f-bar / prod_j (x - lambda-bar_j), low to high (degree d - e_k; {1} when d = e_k)
};
// C-E7: prod_j (x - lambda-bar_j) divides f-bar mod ell with remainder 0, each root at its multiplicity (sequential division);
// ENGINE_CHECK otherwise.  fbar monic, low to high, entries < ell; lam_mod the lambda_j(theta') mod ell.
Attribution AttributeAtEll(const std::vector<uint64_t>& fbar, const std::vector<uint64_t>& lam_mod, uint64_t ell);
// v_ell(a - b) for two exact integers (canonical signed hex); ENGINE_CHECK a != b.
uint32_t ValuationOfDifference(const std::string& a, const std::string& b, uint64_t ell);
// v_ell(c_theta'(lambda)) with c_theta' = f_theta' / prod_j (x - lambdas[j]) over Z (exact division checked); f exact (hex, low to high).
uint32_t CuspResultantValuation(const std::vector<std::string>& f, const std::vector<std::string>& lambdas, size_t j, uint64_t ell);

}  // namespace eisup
}  // namespace heckez

#endif
