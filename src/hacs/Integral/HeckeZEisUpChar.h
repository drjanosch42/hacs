// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

#pragma once
#ifndef INTEGRAL_HECKEZ_EISUPCHAR_H
#define INTEGRAL_HECKEZ_EISUPCHAR_H

// ===========================================================================
// HeckeZEisUpChar -- slot-27 package EISUPb (HZ-EISUP (b)): the Eisenstein U_p systems of the full algebra at level N > 1 where
// e^an_k(N) != 1, i.e. with characters psi != 1 (docs/implementation/slot26/HZEISUPb_design.md v0.4 D1--D8, Sec. 2, 5, 6; [C] 1.28.0-pre.1
// prop:heckeZeisupchar and item 8 (d')(ii) "The components and the maximal ideals from HZ-EISUP (b) on").
//
// WHAT IS COMPUTED, for a `full` record at N > 1 with (C1) certified and e^an_k(N) != 1 (InScopeChar):
//  * C-B0  N cube-free (else ENGINE_CHECK: a defect), s_N = prod_{p odd, p^2 || N} p, o_N = lcm(p - 1 : p | s_N), the characters mod s_N,
//          P^Eis_{q0} = prod over the distinct lambda_psi(T_q0) (q0 = ops[0]) in Z[x] (auxiliary CRT over primes = 1 mod o_N);
//  * C-B1  at word primes q (the full step's retained runs first, then the SAME list by RunAtPrime): K_q = ker P^Eis_{q0}(T_q0) on L / q,
//          skipped unless dim K_q = e_k; the restrictions of every operator of the list to K_q;
//  * C-B2  e_theta = charpoly(theta | E) by CRT under (a)'s bound; e_theta | f_theta exactly over Z, c_theta = f_theta / e_theta;
//  * C-B3  the PROPOSAL: one factor e_{theta,j} = prod_{u in (Z/ord psi)^x} (x - lambda_{psi^u,y}(theta)) per component ([psi], y) of the
//          closed form (Sec. 2.3; the orbit product, the norm from Q(psi)), exactly by auxiliary CRT;
//  * C-B4  e_theta = prod_j e_{theta,j} exactly; sum deg = e_k; (e_theta separable: each factor irreducible by the norm criterion);
//  * C-B5  g_{X,j} = charpoly(X | E_j), E_j = ker e_{theta,j}(theta) | E, for every X of the list, by CRT at the word primes where e_theta is
//          separable, equal to the proposal's orbit product (else ENGINE_CHECK);
//  * C-B6  the identification at q* (the first prime of the word-prime list with q* = 1 mod o_N, q* not in S, p~^{(k-1) o_N} != 1 mod q* for
//          every p~ in S, dim K_q* = e_k, e_theta mod q* separable, the distinct roots of every g_{X,j} pairwise incongruent mod the prime
//          over q* fixed by zeta -> g^{(q*-1)/o_N}, g the least primitive root): the joint system of every line read, psi of every line from
//          the T_p (p in P, p !| N) and the T_p~ (p~ in S, read at q* only), compared with the closed form;
//  * C-B8  (as coded: a decomposition-consistency check, audit m2) at every word prime with e_theta separable: tr(X | K_q) =
//          sum_j tr(X | ker e_{theta,j}(theta) mod q); it holds identically once K_q = (+)_j ker e_{theta,j}(theta).  The design's C-B8 (the
//          exact identity sum_j tr(T_q | E_j) = sum_psi sigma_0(N / cond^2) lambda_psi(T_q) - [k = 2](1 + q^{k-1})) is implied by C-B5.
// Every failure is an ENGINE_CHECK (no record is written); the word-prime cap is the only refusal (WordPrimeCapRefusal).
//
// DETERMINISM.  Fixed prime order, deterministic skip and q*, CRT in index order, exact integers, Galois-invariant outputs only (no Conrey
// label, no embedding in text()).  `reused` / `computed` / `id_prime_extra` are provenance (G10 resp. G9: not in
// record_fnv1a64, but G9 is in cert_sha256, so a change of read policy alone moves cert_sha256; audit m4), never in text().
// ===========================================================================

#include <cstdint>
#include <string>
#include <vector>

#include "HeckeZRational.h"

namespace heckez {
namespace eisup {

// (design D1) the scope of (b): a full record at N > 1, n even, e^an_k(N) != 1 (InScope's complement at N > 1).
bool InScopeChar(uint32_t N, uint32_t n);

// VALIDATION ONLY: tampered inputs (HeckeZEisUpCharValidation arm (g)); each must be caught.  Default: nothing tampered.
struct EisUpCharTamper {
    int32_t ek_delta = 0;            // e_k + delta (C-B1: never satisfied -> the cap refusal on a small cap)
    int32_t factor_plus_one = -1;    // the proposal factor j (in proposal order) with its constant coefficient + 1 (C-B4)
    bool psi_squared = false;        // every proposed orbit built from psi^2 instead of psi (C-B4)
    int32_t y_alpha_at = 0;          // a prime p | N: on every component of degree >= 2 with y_p = 0 and p !| cond psi (a VALID wrong type),
                                     // propose y_p = alpha (C-B4); vacuous (no such component) is an ENGINE_CHECK
    bool norm_full = false;          // the norm from Q(zeta_{o_N}) (the product over (Z/o_N)^x) instead of the orbit product (C-B4)
    bool swap_labels = false;        // the (cond, order) labels of the first two components of different order swapped after C-B6 (C-B6)
    bool forge_U_deg2 = false;       // a degree >= 2 component's g_{U_p} replaced by (x - 1)^deg after C-B5 (C-B5 re-check)
    // (S27 EISUPb audit M2) tampers aimed at the decisive comparisons with cohomology, each leaving C-B4 intact:
    int32_t gprop_plus_one_op = -1;  // the proposal g_{X_o,0} (component 0, operator index o) with its constant coefficient + 1, BEFORE
                                     // C-B5's CRT comparison; e_{theta,j} untouched (C-B4 passes): caught by C-B5's g == gprop
    bool misalign_unweighted = false;   // the closed form of the first degree >= 2 component misaligned on the first T_p whose theta-
                                     // coefficient is 0: the system (psi^u on that T_p, psi elsewhere); theta does not weigh T_p (C-B4
                                     // passes), the orbit marginals are unchanged (C-B5 and the root sets pass): caught by C-B6's
                                     // comparison of the read character with the closed form
    bool qstar_none = false;         // (audit m5) the q* chooser admits no candidate: its own cap refusal (not the e_theta read's)
};

struct EisUpCharInput {
    uint32_t N = 0, n = 0;
    std::vector<uint32_t> ops;       // the operator list: T_p ascending (p !| N), then U_p (p | N) ascending
    uint32_t nU = 0;                 // the last nU entries of ops are U_p
    std::vector<uint32_t> theta;     // theta's coefficients on ops ((C1) certified)
    std::vector<std::string> f;      // f_theta (exact, canonical signed hex, low to high, monic of degree d)
    int64_t e_k = 0, d = 0;
    uint32_t b1 = 0;
    RationalSettings rs;             // threads (fallback runs), word_prime_cap
    const std::vector<PrimeRun>* runs = nullptr;   // the full step's retained runs, may be null
    bool force_recompute = false;    // ignore `runs` (the bit-identity arm)
    bool embed_next_root = false;    // VALIDATION: zeta -> h^{(q*-1)/o_N}, h the NEXT primitive root mod q* (invariance arm)
    std::vector<uint32_t> validation_extra_S;   // VALIDATION ONLY: primes appended to S (exercises the (6') read at q* where S = {})
    bool force_scope = false;        // VALIDATION ONLY: run on an e^an_k(N) = 1 cell (s_N = 1; design Sec. 2.10, arm (j): (b) reproduces (a))
    const EisUpCharTamper* tamper = nullptr;
};

struct EisCharComp {
    uint32_t degree = 0, psi_cond = 0, psi_order = 0;
    std::vector<std::string> theta_poly;               // e_{theta,j} (low to high)
    std::vector<std::string> minpoly;                  // the minimal polynomial of T_q0 on E_j
    std::vector<std::vector<std::string>> Ucharpoly;   // per p | N ascending: g_{U_p,j}
    // degree 1 only: the exact values
    std::string lambda_theta;
    std::vector<std::string> U;
    // internal (never in text(); a representative of the orbit and the type vector, for LambdaPolyOf)
    std::vector<uint32_t> a;                           // exponents of psi on the primitive roots of the p | s_N
    std::vector<int32_t> y;                            // per p | N ascending: 0 (U_p = 0), 1 (alpha = psi(p)), 2 (beta = psi-bar(p) p^{k-1})
};

struct EisUpCharResult {
    uint32_t N = 0, n = 0, nU = 0;
    std::vector<uint32_t> ops, theta;
    int64_t e_k = 0, d = 0;
    uint32_t q0 = 0, s_N = 0, o_N = 0;
    std::vector<uint32_t> sp, gen;                     // the primes of s_N ascending and their least primitive roots (internal)
    std::vector<std::string> P_eis;                    // P^Eis_{q0} (low to high)
    std::vector<std::string> e_theta;                  // charpoly(theta | E)
    std::vector<EisCharComp> comps;                    // the record's row order (degree 1 first, then degree >= 2)
    std::vector<std::string> c_theta;                  // f_theta / e_theta
    uint32_t primes_used = 0;                          // word primes read for e_theta (dim K_q = e_k)
    uint32_t sep_used = 0;                             // of those and further ones: the primes read for the g_{X,j}
    std::vector<uint32_t> skipped;                     // word primes with dim K_q != e_k, in list order (up to the last read)
    uint32_t id_prime = 0;                             // q*
    std::vector<uint32_t> S;                           // the extra primes p~ (Sec. 2.8)
    // provenance (not in text())
    uint32_t id_prime_extra = 0;                       // 1 iff q* needed a run of its own
    uint32_t reused = 0, computed = 0;
    double secs = 0;

    std::vector<uint32_t> Uprimes() const;
    std::string text() const;
    uint64_t digest() const;
};

// The route.  Requires InScopeChar(N, n), nU >= 1, consistent inputs (ENGINE_CHECK); throws WordPrimeCapRefusal beyond the cap.
EisUpCharResult EisUpCharStep(const EisUpCharInput& in);

// charpoly(theta' | E_j) exactly (canonical signed hex, low to high) for theta' = sum c_i X_i over `thP` (p !| N: T_p, acting on a line of
// character psi by lambda_psi(T_p), T_ell included (prop:heckeZeisupchar (7)); p | N: U_p by the component's type); the orbit product.
std::vector<std::string> LambdaPolyOf(const EisUpCharResult& R, size_t j, const std::vector<uint32_t>& thP, const std::vector<uint32_t>& coeffs);

// v_ell Res(A, B) of two monic integer polynomials (canonical hex, low to high), exact over Z (Bareiss); ENGINE_CHECK Res != 0.
uint32_t ResultantValuation(const std::vector<std::string>& A, const std::vector<std::string>& B, uint64_t ell);
// v_ell(disc f) = v_ell Res(f, f') for monic f (exact; the validation's design-time agreement arm).
uint32_t DiscValuation(const std::vector<std::string>& f, uint64_t ell);
// A / B over Z for monic B, exact (ENGINE_CHECK remainder 0).
std::vector<std::string> PolyDivExact(const std::vector<std::string>& A, const std::vector<std::string>& B);
// residues mod ell (< 2^63) of an exact polynomial
std::vector<uint64_t> PolyModEll(const std::vector<std::string>& A, uint64_t ell);

// (validation arms) the closed form alone: the multiset of components (degree, psi_cond, psi_order, y as text) of E_k(N), and its count
// sum_psi sigma_0(N / cond^2) - [k = 2]; and the chooser of S and of q* on an arithmetic level (unit checks of arm (f')).
struct ClosedFormSummary {
    std::vector<std::string> comps;   // "deg=<D> cond=<c> ord=<o> y=<types>" sorted
    int64_t systems = 0;              // sum of the degrees
    int64_t cusps_minus = 0;          // c(N) - [k = 2]
};
ClosedFormSummary ClosedForm(uint32_t N, uint32_t k);
std::vector<uint32_t> ExtraPrimesS(uint32_t N, const std::vector<uint32_t>& Tprimes);
// the arithmetic conditions on q* (q = 1 mod o_N, q not in S, p~^{(k-1) o_N} != 1 mod q for p~ in S)
bool IdPrimeArithmeticOk(uint32_t q, uint32_t o_N, uint32_t k, const std::vector<uint32_t>& S);
// the direction check of Sec. 2.7 at (121, 0) (o_N = 10): psi(2) = zeta^t read under the embedding sigma_c: returns the exponent read
// at the conjugate prime, i.e. (c * t) mod o_N (sigma_c o lambda) -- computed through residues at a split prime, not by the formula.
uint32_t DirectionRead(uint32_t o_N, uint32_t t, uint32_t c, uint32_t q);

}  // namespace eisup
}  // namespace heckez

#endif
