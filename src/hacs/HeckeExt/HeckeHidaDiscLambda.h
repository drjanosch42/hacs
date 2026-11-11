// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

#pragma once
#ifndef HECKE_EXT_HECKE_HIDA_DISC_LAMBDA_H
#define HECKE_EXT_HECKE_HIDA_DISC_LAMBDA_H

// ===========================================================================
// (hecke-extensions slot 26, package Z6 = H2) THE DISCRIMINANT SERIES OF THE RAMIFIED
// ORDINARY FAMILIES AT TAME LEVEL 1.  Binding text: [U] docs/latex/hida_hecke_algebra.tex
// 0.7.4-pre.1 part 7 (sec:u-H2): def:u-DF (D_F = tau^2 - 4 nu of a generator), lem:u-DF
// (the eleven blocks of prop:u-loewy(4) are regular rank-2 families; D_F in m \ m^2),
// lem:u-DFread (what D_F mod (ell^K, Z^m) decides: case (A) = unit linear coefficient, rung
// sub-tower k = s_F mod ell; case (B) = ramified at every weight), pr:u-H2 (U25).  Design:
// docs/implementation/slot26/Z6_design.md v0.2 (audit Z6_design_audit.md); report Z6_report.md.
//
// The new piece is M3 of 05 Sec. 4.2: the PER-BLOCK Fitting split at a NON-Eisenstein residual
// system, by sequential generalised kernels of T_p - lambda_p (p ascending), then U - u, on
// e S_k (the forms side of HeckeHidaLambda, ComputeFormsWeight).  Deterministic: no random linear
// combination.  Certificate: rank 2 (= rank_Lambda h(a)_m, lem:u-DF(1)) and every T_p - lambda_p,
// U - u nilpotent mod ell on the block.  At the rung weights the block is cut with the short list
// P' = SturmPrimes(ell, k0) (subset of {2, 3, 5, 7}): the m-part lies in the generalised eigenspace
// of any sub-system and has rank 2 by control, so rank 2 forces equality; a collision shows as
// rank > 2 and is REFUSED (ENGINE_CHECK), never mis-read.
//
// The Lambda stage is thm:u-interp (InterpolateTruncation) on the five values
// Delta(x_{k_j}) mod ell^5, j = 0..4, K = m = 3, centred at x_{k0}; then the reading of
// lem:u-DFread(R).  No new arithmetic: pure host C++, mod ell^W < 2^84.
//
// Writes nothing, no golden object, no record, no setting, no digest of record.  Every refusal is
// an ENGINE_CHECK (std::logic_error).
//
// PARALLEL DRIVER (ComputeFormsParallel): the per-weight jobs ComputeFormsWeight run concurrently
// on oc_util::ParallelForBlocks (std::thread, dynamic claiming of one job at a time), each writing
// only its own slot of a pre-sized vector (an index-addressed merge); task exceptions are captured
// and the first one in job order is rethrown on the caller.  The result is bit-identical at any
// thread count.  ComputeFormsWeight carries oc_util::CancelPoints, which CancelPoint.h forbids
// inside a parallel-region body: the hook is null in validations, so nothing breaks, but THIS
// DRIVER MUST NEVER BE BOUND TO A CONSOLE VERB (audit F12).
// ===========================================================================

#include <cstdint>
#include <map>
#include <string>
#include <vector>

#include "HeckeHidaLambda.h"

namespace hecke_ext {
namespace disclambda {

using lambda::Mat;
using lambda::u128;
using lambda::Vec;

// A residual system (lambda_bar_p)_{p in P}, u_bar at a block of e S_k (lem:u-DF(1): the same at every weight of the class).
struct ResidualSystem {
    uint64_t ell = 0;
    uint32_t k = 0;                   // the weight it was found at
    std::vector<std::string> ops;     // "T2", "T3", ..., "U" (the cut order: T_p ascending, U last)
    std::vector<uint64_t> res;        // residues mod ell, parallel to ops
    uint32_t rank = 0;                // rank of the generalised eigenspace on e S_k
    uint64_t at(const std::string& op) const;   // ENGINE_CHECK: op present
};

// The search of step 2 (design Sec. 2.1): every root of chi(T_2 | e S_k) mod ell of multiplicity >= 2, split further by the
// generalised eigenspaces of T_p (p in primes \ {2}, ascending) and U with eigenvalues in F_ell.  A piece of rank 1 (the double
// root separated) or one without F_ell-rational eigenvalues (a conjugate pair over F_{ell^d}, the (54, 67), (46, 383) type) is a
// reduced coincidence and REFUSED; so is a piece of rank >= 2 on which every operator is scalar mod ell.  The non-reduced
// systems are returned in ascending lexicographic order of their residues (T2 first).
struct SystemSearch {
    std::vector<ResidualSystem> systems;
    uint32_t candidates = 0;          // T_2 roots of multiplicity >= 2
    uint32_t refused = 0;             // pieces refused as reduced coincidences
    std::vector<std::string> notes;   // why, one per refusal
};
SystemSearch FindNonReducedSystems(const lambda::FormsWeight& w, const std::vector<uint64_t>& primes);

// The block of e S_k at the residual system `sys`, cut by the sequential generalised kernels of T_p - lambda_p (p in primes,
// ascending; every p must be in sys) and then U - u.  REFUSED (ENGINE_CHECK) unless rank 2 and every cut operator minus its
// residue is nilpotent mod ell on the block.  ops: every operator of the forms data (w.S.names) restricted to the block, mod ell^W.
struct BlockAt {
    uint32_t k = 0, rank = 0;
    std::map<std::string, Mat> ops;
};
BlockAt ResidualBlock(const lambda::FormsWeight& w, const std::vector<uint64_t>& primes, const ResidualSystem& sys);

// The second generator t' of step 4: "T<q'>" for the first q' in primes \ {2} with T_q' - lambda_q' != 0 mod ell on the block
// (GeneratesModEll), else "T2+T<q'>" with q' the first prime of primes \ {2} (audit Q10).
std::string SecondGenerator(const BlockAt& b, const std::vector<uint64_t>& primes, uint64_t ell);
// The matrix of a generator spec ("T<q>" or "T<a>+T<b>") on the block, mod ell^W.
Mat GeneratorMatrix(const BlockAt& b, const std::string& spec, const lambda::ZMod& R);

// One weight's reading of a block (step 4): chi(t | V) = Y^2 - tau Y + nu mod ell^W, Delta = tau^2 - 4 nu.
struct Quad {
    u128 tau = 0, nu = 0, disc = 0;
    uint32_t v_disc = 0;              // v_ell(disc), capped at W
    bool gen = false;                 // t generates mod ell (t - c != 0 mod ell)
};
Quad ReadQuad(const Mat& t, const lambda::ZMod& R);

// The Lambda stage of one block (design Sec. 2.2, 2.3) from the values Delta(x_{k_j}) mod ell^5 at k_j = k0 + j(ell - 1),
// j = 0 .. N-1, N = 5 or 6 (the sixth is the held-out weight).  K = m = 3.
struct DiscTruncation {
    Vec e;                            // e0, e1, e2: D mod (ell^3, Z^3), Z = X - x_{k0}
    uint32_t v_e0 = 0;
    Vec sub;                          // the (2, 2) truncation from j = 2, 3, 4 (centre x_{k0}), mod ell^2
    bool subset_ok = false;           // sub == e mod (ell^2, Z^2): a code self-check, NOT an error detector -- it holds for any
                                      // values once the five-node interpolant is ell-integral (Z6_code_audit.md F1)
    int heldout = -1;                 // -1 skip (N = 5), 0 failed, 1 held: r(x_{k5}) == Delta(x_{k5}) mod ell^5
    char cas = '?';                   // 'A' (e1 a unit) or 'B'
    u128 zeta = 0, xi = 0;            // case A: the root of e0 + e1 Z + e2 Z^2 in ell Z_ell mod ell^3, xi = x_{k0} + zeta mod ell^3
    uint64_t sF_mod_l2 = 0;           // case A: u^{s_F} - 1 == xi mod ell^3
    uint64_t rung2 = 0, rung3 = 0;    // case A: the least class weights >= 3 with k == s_F mod ell, resp. ell^2
    std::string lambda_word;          // "1" (A), "2" (B, e2 a unit), "undecided" (B otherwise)
    bool reading_ok = false;          // the internal v-reading at the nodes (a code check)
};
DiscTruncation ReadDisc(uint64_t ell, uint32_t k0, const Vec& disc_values);
// The predicted v(D(x_k)) of lem:u-DFread at a weight k: case A 1 + v_ell(k - s_F) with s_F known mod ell^2 (returns 3 for
// ">= 3"), case B 1.
uint32_t PredictedV(const DiscTruncation& d, uint64_t ell, uint64_t k);
// The (K, m) truncation of a single value series (tau or nu) as InterpolateTruncation (Kp = 5, centre k0) on the first five.
Vec Trunc33(uint64_t ell, uint32_t k0, const Vec& values);
// Ring consistency (a code check): tau~^2 - 4 nu~ == D~ mod (ell^3, Z^3).
bool RingConsistent(uint64_t ell, const Vec& tau3, const Vec& nu3, const Vec& D3);
// q-independence's one bit: (e0'/ell) / (e0/ell) mod ell is a non-zero square (both v(e0) = 1; false otherwise).
bool UnitSquareRatio(uint64_t ell, u128 e0, u128 e0p);

// x_k = (1 + ell)^k - 1 mod R.M.
u128 XMod(uint64_t ell, uint64_t k, const lambda::ZMod& R);

// The parallel driver (see the header comment; never bound to a console verb).
struct FormsJob {
    uint64_t ell = 0;
    uint32_t k = 0, W = 0;
    std::vector<uint64_t> primes;
};
std::vector<lambda::FormsWeight> ComputeFormsParallel(const std::vector<FormsJob>& jobs, unsigned threads);

}  // namespace disclambda
}  // namespace hecke_ext

#endif  // HECKE_EXT_HECKE_HIDA_DISC_LAMBDA_H
