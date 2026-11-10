// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

#pragma once
#ifndef INTEGRAL_DESCENT_DESCENT_FREE_BASIS_Z_H
#define INTEGRAL_DESCENT_DESCENT_FREE_BASIS_Z_H

// ===========================================================================
// EXPORT-Z step 2, package EXZ2-P1b (hecke-extensions slot 25; docs/implementation/slot25/EXPORTZ2_design.md v2 Sec. 1.2-1.4, 7.3;
// audits EXPORTZ2_design_audit.md F1-F3, F13 and EXZ2P1a_audit.md F1, F2, F5, F10).  The free block's BASIS: from the K Dixon pairs
// (B_s, Phi_s) of IsoTreeCarrierZ.h (rule iso_dixon/v3) to a Z-basis B of L = H^1(Gamma_0(N); Sym^n Z^2) / torsion and a Z-basis
// Phi_cert of L^dual, proved by ONE determinant.
//
// Sec. 1.2 LATTICE SUMS (LatticeSums).  X = Phi_0^T [B_0 | ... | B_{K-1}] (r x Kr, class coordinates: Phi_0^T kills B^1 and is injective
// on L).  Column HNF with transform by incremental insertion (the columns in the order s = 0 .. K-1, j = 0 .. r-1; slot i holds the
// basis vector with pivot at row i and zeros below; a non-divisible pivot is combined by the 2 x 2 unimodular Bezout step
// [[s, -b/g], [t, a/g]]; off-diagonals reduced into [0, h_ii) after the last insertion).  The inserted vectors that reduce to 0 are a
// Z-basis of the relation lattice ker X (the transform is unimodular); the entries above a pivot are kept as symmetric remainders during
// the insertion.  Size reduction of the transform modulo that lattice in the norm weighted by 4^{bits of the cochain column}: the
// relation lattice LLL-reduced (exact integral LLL, nf::lll_basis, delta 3/4) and Babai's nearest plane (exact rational Gram-Schmidt)
// on every transform column, when there are at most lll_max_relations (= 48) relations; above that (pinned; the exact LLL is too slow
// there -- open item O-7 for P3: a fast integer LLL) only the reduction by the STRUCTURAL relations of the diagonal first block
// (rho_t = d_t e_t - G_0^{-1} d_t x_t: every coordinate t >= r into (-d_t/2, d_t/2]).  B_Sigma = [B_s] U.
// (Fix round 1, audit EXZ2P1b_audit.md F1.)  On the structural path the same relations reduce the transform INSIDE HnfInsert, after
// every step, so no transform entry exceeds ~ bits(d_t) + bits(class) (P1b: 313 816 bits at (23, 28), 1098 s).  Bit-identical to
// the post-hoc reduction: every coefficient HnfInsert uses (quotients, Bezout s, t, the remainders above a pivot, the sign flips) is a
// function of the CLASS vectors only, which a relation does not change, so the transform changes by an element of the Z-span of the
// rho_t; the final StructuralReduce is a function of u_t mod d_t (t >= r) and of the class, hence canonical modulo that span.  The
// LLL path is unchanged (its input basis, the relation vectors, would change).  The sums' memory is predicted (PredictSumsRss) and
// refused [RssCeiling] before the HNF, against the same ceiling as the pairs.
// Likewise Phi_Sigma from Y = [Phi_s]^T B_0 (the functionals in the coordinates of B_0, a Q-basis of L; Phi_0^T B_0 = G_0 is diagonal,
// CHECKED).  Delta_Sigma = |det(Phi_Sigma^T B_Sigma)| = prod h_B,ii prod h_Phi,ii / prod g_i (exact, CHECKED, and confirmed mod two
// primes); G_Sigma itself is not triangular (the design's "G_Sigma is triangular" holds for the Y = [Phi_s]^T B_Sigma variant, whose
// first block is dense in this insertion order and whose transform reached thousands of bits on the measurement cells).
//
// Sec. 1.3 SATURATION (Saturate) on the carrier, side B (X = B, A = d^0) before side Phi (X = Phi, A = (d^1)^T), never densified:
//   * one sparse local Smith pass per (side, modulus) mod M: right-looking elimination with valuation-ordered passes v = 0 .. k-1
//     (column order ascending; pivot = the non-pivoted row with an entry of valuation exactly v and the fewest non-zeros, ties by row
//     index), row operations recorded (U), the pivot row's other entries cleared by column operations (recorded for the
//     back-substitution), so U A V = diag(u_c ell^{v_c}) on the pivot rows and 0 elsewhere mod M;
//   * per round: the residuals (U X)_{non-pivot rows} mod `unit`, their dependencies by an echelon in insertion order (= the rref's
//     free columns; c_f = 1, zero on the other free columns), and for ALL kernel vectors at once y~ with A y~ = unit^sigma X c
//     (mod M = unit^{sigma+1}), x_f <- (unit^sigma X c - A y~) / M (exact division, CHECKED).  The index drops by unit^k per round
//     (k = dim ker) and is TRACKED exactly (det is multilinear and c_f = 1); rounds stop at an empty kernel.
//   * pieces of |Delta| (no factorisation): (i) the record's torsion primes (hecke3::RecordTorsionPrimes, with 2, 3) and (ii) the primes
//     < 2^20 found by trial division: prime-power ("local") mode, M = ell^k with k = 1, 2, 4, ... until the Smith pass has the rational
//     rank (k - 1 >= the ell-exponent of tors coker A, i.e. v_ell(exp T) on side B, v_ell(exp tors H^2) on side Phi -- adaptive, so no
//     step-1 export is needed; sigma = k - 1 affects progress only); (iii) a cofactor < 2^62: composite mode (Z/c treated as a field,
//     a non-unit exposes g = gcd and splits c, both factors recurse); (iv) a cofactor >= 2^62: K <- K + 1 (<= 5, the v3 orders
//     s = 3, 4) and Sec. 1.2 again; (v) at K = 5: composite mode over the MULTI-WORD modulus (nf::BigInt residues; P1a audit F2);
//     only then the resource refusal `SaturationCofactor` (multiword disabled, the cofactor above multiword_max_bits, or
//     max_iter iterations).  Order (pinned): pieces ascending, the cofactor last; side B before side Phi; repeat until |Delta| = 1.
//     (Fix round 1, F4) an iteration that replaces no column is deterministic and would repeat: refused at once, `SaturationStall`.
//     (F8) composite pieces re-test gcd(Delta, m) before the side-Phi pass.
//
// Sec. 1.4 THE CERTIFICATE (CertifyBasis): d^1 B = 0 and Phi^T d^0 = 0 exactly (nf::BigInt), and det(Phi^T B) = +-1 by residues at
// CrtPrimes primes whose product exceeds 2 x the Hadamard bound (exact sum of squares per row): |det(Phi^T B)| = [L : B] [L^dual : Phi]
// proves that B is a Z-basis of L and Phi of L^dual at every prime at once, GIVEN rank L = r (audit Sec. 1: with rank L = R > r it
// would only prove a saturated rank-r summand).  (Fix round 1, F6) that premise is proved inside CertifyBasis, not taken from the
// closed-form b_1: a witness prime q_w (the first CrtPrimes prime != skip_q prime to 6N; the next ones on a deficit, at most 8) with
// rank_{q_w} d^1 + rank_{q_w} d^0 >= dim C^1 - r gives rank L = dim C^1 - rank_Q d^1 - rank_Q d^0 <= r (ranks mod q are <= the
// rational ones), and det != 0 gives rank L >= r.  Printed: saturation = "det_pm1" (only when the certificate holds), basis_rule =
// "iso_dixon/v3" with q0 / q0_confirm / q0_skipped, basis_fnv = PackedFnv(B), phi_fnv = PackedFnv(Phi_cert).  The tracked index is never trusted: the certificate is
// computed from scratch, and a tracked 1 with a failing certificate is a defect (ENGINE_CHECK).
//
// Deterministic throughout (no seeds, no floating point in a decision); single-threaded (the lanes of P2 are the parallel axis).
// No console surface, no record key, no golden object, no settings key (P3).
// ===========================================================================

#include <cstdint>
#include <string>
#include <vector>

#include "IsoTreeCarrierZ.h"

namespace integral_descent {
namespace freez {

struct FreeBasisOptions {
    PairsOptions pairs;                   // K = 3 to start (iso_dixon/v3)
    uint64_t sums_rss_cap_bytes = 0;      // the sums' RssCeiling (fix round 1, F1 (b)); 0 = pairs.rss_cap_bytes (one ceiling, O-3)
    uint32_t K_max = kPairsKMax;          // escalation (iv)
    uint32_t max_iter = 20;               // whole Sec. 1.3 repetitions
    bool size_reduce = true;
    uint32_t lll_max_relations = 48;      // LLL + Babai when the relation lattice has at most this many vectors; above: the structural one
    bool multiword = true;                // (v): the multi-word composite mode before SaturationCofactor
    uint32_t multiword_max_bits = 1u << 16;
    uint32_t local_max_k = 62;            // ell^k < 2^62
    bool verbose = false;                 // one EXZ2_P1B_PHASE line per phase on stdout (measurement)
};

namespace testing {
// Validation-only hooks (fix round 1, audit F3 / F10 / F1: not in the public options).  nullptr = production.
struct FreeBasisHooks {
    bool sums_posthoc_only = false;       // Sec. 1.2: the structural reduction only after the HNF (the P1b route; the identity arm)
    uint32_t cofactor_limit_bits = 62;    // Sec. 1.3 (iii) / (iv): composite64 below 2^this, else escalation / multi-word
    uint32_t trial_bound = 1u << 20;      // Sec. 1.3 (ii): trial division by the primes below this
    PairsHooks pairs;
};
}  // namespace testing

struct SatStep {
    char side = 'B';                      // 'B' or 'P' (Phi)
    std::string piece;                    // "2", "3", "c31b", ...
    std::string mode;                     // local / composite64 / multiword
    uint32_t k = 1;                       // local: M = ell^k
    uint32_t rounds = 0, cols = 0;        // kernel rounds, replaced columns
    uint32_t smith_pivots = 0;
    uint64_t smith_ops = 0, smith_nnz = 0;
    double seconds = 0;
};

struct SumsResult {
    PackedZ B, Phi;
    BigInt delta;                         // Delta_Sigma = |det(Phi_Sigma^T B_Sigma)| = prod h_B,ii prod h_Phi,ii / prod g_i
    uint32_t K = 0, relations_B = 0, relations_Phi = 0;
    uint32_t bits_U0 = 0, bits_U = 0, bits_V0 = 0, bits_V = 0;   // transform bits at the end of the HNF / after the size reduction
    uint32_t bits_U_peak = 0, bits_V_peak = 0;                   // the largest transform entry DURING the insertion (fix round 1, F1)
    bool lll_B = false, lll_Phi = false;  // LLL + Babai (<= lll_max_relations relations); false: the structural reduction
    bool inline_B = false, inline_Phi = false;   // the structural reduction ran inside the insertion (fix round 1, F1)
    uint64_t rss_pred = 0;                // PredictSumsRss (bytes)
    uint32_t pred_bits_x = 0, pred_bits_g = 0, pred_bits_d = 0;   // its inputs: max bits of x / y, of g_i, of d_t (structural path)
    bool refused = false;                 // (fix round 1, F1 (b)) REFUSED [RssCeiling] before the HNF
    std::string refusal;
    double seconds = 0;
};
// Sec. 1.2 on pairs[0 .. K-1].
SumsResult LatticeSums(const IsoTreeCarrierZ& c, const std::vector<Pair>& pairs, const FreeBasisOptions& opt = {},
                       const testing::FreeBasisHooks* hooks = nullptr);
// (Fix round 1, audit F1 (b)) the predicted transient bytes of LatticeSums on these pairs (decided before the HNF; a posteriori in
// the pairs, a priori in the HNF): per BigInt entry 32 B (header) + h(b) (the heap block, the formula of PredictPairsRss), over the
// class / value vectors (K r x r per side), the HNF basis (r x r), the transforms (r x K r) and the relation vectors (stored on the
// LLL path only, (K - 1) r x K r), the LLL state (3 R (R + K r) rationals of 2 (b_U + b_cols) + 64 bits), the structural data,
// plus the combined columns (PackedZ, nnz <= r min(dim C^1, the parts' nnz)) and the map of Combine; x 1.25, + 1 MiB.  Bits: class entries
// max(bits x, max bits g_i) + 64 (h_ii | g_i); transform entries b_U = max bits d_t + bits x + max bits g_i + 64 on the structural
// path (inline reduction), 64 + r (bits x + max bits g_i) on the LLL path (r <= 24; P1b's measured unreduced growth).  Not a proof:
// a calibrated estimate, compared with the sampled growth in arm (M) (never below; EXZ2P1b_report.md, fix round 1).  The ceiling
// test is 16 MiB + PredictSumsRss + the pairs' resident bytes <= the ceiling (sums_rss_cap_bytes, default pairs.rss_cap_bytes).
uint64_t PredictSumsRss(const IsoTreeCarrierZ& c, const std::vector<Pair>& pairs, const FreeBasisOptions& opt = {});

struct SaturationResult {
    bool done = false;                    // the tracked |Delta| reached 1
    bool refused = false;
    std::string refusal;                  // "REFUSED [SaturationCofactor] ..." / "REFUSED [SaturationStall] ..." (fix round 1, F4)
    bool stalled = false;                 // an iteration replaced no column (refused at once)
    bool wants_escalation = false;        // (iv): a cofactor >= 2^62 at K < K_max (the caller escalates)
    uint32_t iterations = 0, splits = 0;
    uint32_t multiword_passes = 0;        // multi-word (RingBig) passes started, including those ended by a gcd split
    BigInt delta;                         // the tracked |det(Phi^T B)| at the end
    std::vector<SatStep> log;
    double seconds = 0;
};
// Sec. 1.3 in place on (B, Phi) from the exact |det(Phi^T B)| = delta.  escalate_allowed: report a cofactor >= 2^62 as
// wants_escalation instead of entering the multi-word mode.
SaturationResult Saturate(const IsoTreeCarrierZ& c, PackedZ& B, PackedZ& Phi, BigInt delta, const FreeBasisOptions& opt,
                          bool escalate_allowed, const testing::FreeBasisHooks* hooks = nullptr);

struct Certificate {
    bool ok = false;
    bool cocycles = false, kills_d0 = false;
    int sign = 0;                         // det = sign when ok
    uint32_t primes = 0, hadamard_bits = 0;
    // (fix round 1, F6) the rank premise rank L <= r, proved at a witness prime
    bool rank_ok = false;
    uint32_t rank_q = 0, rank_d1 = 0, rank_d0 = 0, rank_tries = 0;
    double seconds = 0;
    // (Z3a fix1, audit F2) what this certificate is FOR, stored by CertifyBasis itself: the fingerprints of (B, Phi) and the carrier key
    // (N, n, dim C^0, C^1, C^2), so a reuse (FreeBlockInput::PriorCert) cannot be given loose, mismatched fields
    uint64_t fnv_B = 0, fnv_Phi = 0;
    uint32_t car_N = 0, car_n = 0, car_dim0 = 0, car_dim1 = 0, car_dim2 = 0;
};
// skip_q: a prime not used as the rank witness (BuildFreeBasis passes its q0; the witness is independent of the pairs' prime)
Certificate CertifyBasis(const IsoTreeCarrierZ& c, const PackedZ& B, const PackedZ& Phi, uint32_t skip_q = kQ0);

struct FreeBasisResult {
    bool refused = false;
    std::string refusal;
    PairsResult pairs;
    SumsResult sums0;                     // the first Sec. 1.2 (K = 3): Delta_Sigma of record
    SumsResult sums;                      // the last one (after an escalation)
    SaturationResult sat;
    Certificate cert;
    PackedZ B, Phi_cert;
    uint32_t K_used = 0;
    uint64_t fnv_B = 0, fnv_Phi = 0;
    uint32_t bits_B = 0, bits_Phi = 0;
    std::string saturation;               // "det_pm1" only when the certificate holds (fix round 1, F9); empty otherwise
    std::string basis_rule = kBasisRule;  // printed with pairs.q0, pairs.q0_confirm, pairs.q0_skipped (F7)
    double t_pairs = 0, t_sums = 0, t_sat = 0, t_cert = 0, seconds = 0;
};
FreeBasisResult BuildFreeBasis(const IsoTreeCarrierZ& c, const FreeBasisOptions& opt = {}, const testing::FreeBasisHooks* hooks = nullptr);
// "q0=<q0> q0_confirm=<q1> q0_skipped=<a/b/..|->" (printed beside basis_rule, F7)
std::string Q0Key(const PairsResult& P);

// helpers (exposed for the validation and the report)
std::string FactorString(const BigInt& x);                       // "2^4*3^2*c123b" (trial division < 2^20, cofactor by its bits)
std::vector<uint32_t> SmallPrimes(uint32_t below);               // the primes < below (sieve)
uint32_t TransformBits(const std::vector<std::vector<BigInt>>& U);
// det mod a word prime (Gaussian elimination, row-major r x r residues)
uint32_t DetModPrime(std::vector<uint32_t> A, uint32_t r, uint32_t p);

}  // namespace freez
}  // namespace integral_descent

#endif
