// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

#pragma once
#ifndef HECKE_ORDINARY_PART_H
#define HECKE_ORDINARY_PART_H

// ===========================================================================
// (hecke-extensions slot 12, WP-4) THE CERTIFIED ORDINARY PROJECTOR IN CODE.
// Binding text: [C] v1.12.0-pre.2 sec:hida -- prop:torsionprepass (the torsion
// pre-pass), cor:ordprecision (the certified precision, before the run),
// lem:ordexponent (the ordinary projector as one power of U),
// rmk:ordexponentuse (which generator count and precision to use), and the
// predictions (Hi1), (Hi2) of rmk:hidacheckable, which
// HeckeOrdinaryPrecisionValidation tests against slot-11 WP-18c's LadicCell.
//
// WHAT IT IS.  Pipeline-free, value-free with respect to every shipped object:
// nothing here is reached from an existing record, report, campaign job id or
// golden text.  LadicCell reaches it only under the OPT-IN
// Request::precision_source = Prepass (the default K-mode route is unchanged).
//
//  * TorsionExponentPrepass(n, ell) -- prop:torsionprepass(2)-(3): J_n =
//    [rho_n(S) - 1 ; rho_n(T) - 1] (2(n+1) x (n+1), ComputeSymmetricPowerMatrixRing,
//    the pipeline's basis and convention), lambda_e = length ker(J_n mod ell^e)
//    for e = 1, 2, ... (e = 1 over F_ell, e >= 2 by engine::elim::kernel_module),
//    stopped at the first e with lambda_{e+1} = lambda_e: that e IS e_max.  The
//    torsion multiset {a_t} is read independently from the Smith rule
//    (engine::elim::invariant_factors of colspan J_n over Z/ell^{e_max+1}, whose
//    full rank r_{e_max+1}(J_n) = n+1 is the second form of the stopping rule).
//    Valid at EVERY prime, n >= 1 (clause (6)); n = 0 is refused by name.
//  * CertifiedWorkingPrecision(n, ell, e') -- cor:ordprecision(2): K = e' + e_max,
//    ell >= 5 only (clause (6): at ell in {2, 3} the pre-pass is TRUE but is not
//    a precision for the model -- refused by name, ENGINE_CHECK_MSG, exit 3).
//  * OrdinaryExponent -- N_ord(g, K) = ell^{K-1+c} prod_{f<=g}(ell^f - 1),
//    c = ceil(log_ell g), carried as its factors (never materialised; the
//    u128 value only when it fits, for the oracle).
//  * RingMat / OrdinaryProjector -- an endomorphism of (+)_i Z/ell^{a_i}
//    (column convention A b_j = sum_i A_ij b_i, row i meaningful mod ell^{a_i};
//    the shipped H_matrix_module convention) and U^N by the factors of N.
// ===========================================================================

#include <string>
#include <cstdint>
#include <vector>

namespace hecke {
namespace ordinary {

using u128 = unsigned __int128;

// ---- prop:torsionprepass ----------------------------------------------------
struct TorsionPrepass {
    uint32_t n_sym = 0;
    uint64_t ell = 0;
    uint32_t e_max = 0;
    std::vector<uint32_t> lambda;       // lambda[e], e = 0 .. e_max + 1 (lambda[0] = 0)
    std::vector<uint32_t> torsion;      // the a_t, descending (Smith rule at Z/ell^{e_max+1})
    uint32_t rank_full_level = 0;       // r_{e_max+1}(J_n) (== n + 1: the second stopping rule)
};
// Any prime ell, n >= 1.  Throws std::logic_error (ENGINE_CHECK_MSG naming
// prop:torsionprepass) at n = 0.  A level ell^{e+1} that does not fit uint64_t
// runs on the engine::ladic digit route since slot-13 WP-5 (PrepassRoute below);
// NotImplementedInThisBuild only where that route cannot run (ell >= 2^31).
TorsionPrepass TorsionExponentPrepass(uint32_t n_sym, uint64_t ell);

// (slot-13 WP-5) The pre-pass beyond 2^64: the ROUTE of the lambda_e / Smith
// computations.  Auto (the default, TorsionExponentPrepass above) is the native
// uint64_t route at every level ell^e that fits -- byte for byte the slot-12 code
// -- and the engine::ladic digit route (LadicInduction.h, W = uint64_t) at every
// level that does not (it raised NotImplementedInThisBuild until slot 13); ell
// must then satisfy ell < 2^31 (the digit layout's bound) and ell >= 2^32 stays
// refused by name.  Native / Ladic force one route at EVERY level e >= 2 (the
// oracle arm; e = 1 is over F_ell on both).  digit_exp_override (TEST-ONLY, the
// digit layout's m' of ENGINE_API.md Sec. 13.11.2): 0 = the word-maximal m;
// m' = 1 makes L = e planes, so the level induction really runs -- it changes the
// layout, never a value.  J_n's entries on the digit route are the exact
// integers (rho_n(T): binomials by Pascal's rule, rho_n(S): +-1) as base-ell digits.
enum class PrepassRoute : uint32_t { Auto = 0, Native = 1, Ladic = 2 };
struct PrepassOptions {
    PrepassRoute route = PrepassRoute::Auto;
    uint32_t digit_exp_override = 0;
};
TorsionPrepass TorsionExponentPrepass(uint32_t n_sym, uint64_t ell, const PrepassOptions& opt);
// lambda_e = length ker(J_n mod ell^e) by the given route (e >= 1), and the torsion
// multiset read by the Smith rule at level E (E >= 2): both at ANY level, e.g.
// beyond 2^64 on the digit route (the stationarity lambda_e = lambda_{e_max} for
// e > e_max, and the level-independence of the Smith reading, are the oracle).
uint32_t PrepassKernelLength(uint32_t n_sym, uint64_t ell, uint32_t e, const PrepassOptions& opt);
std::vector<uint32_t> PrepassTorsionAtLevel(uint32_t n_sym, uint64_t ell, uint32_t E, const PrepassOptions& opt);
// J_n mod ell^E as base-ell digits (E digits per entry, little-endian), row-major
// 2(n+1) x (n+1): the digit route's matrix, from the exact integers (any E >= 1).
std::vector<std::vector<uint32_t>> PrepassMatrixDigits(uint32_t n_sym, uint64_t ell, uint32_t E);

// J_n mod ell^E, row-major 2(n+1) x (n+1) (E >= 2) -- exposed for the oracles.
std::vector<uint64_t> PrepassMatrix(uint32_t n_sym, uint64_t ell, uint32_t E);
// A generating set of ker(J_n mod ell^e) (rows, length n+1), e >= 1.
std::vector<std::vector<uint64_t>> PrepassKernelGenerators(uint32_t n_sym, uint64_t ell, uint32_t e);
// T_p on H^0(Gamma; V/ell^e) = ker(J_n mod ell^e), directly: v -> sum_beta rho_n(beta) v
// over the RIGHT coset representatives beta of Gamma diag(1, p) Gamma
// ([[p, b], [0, 1]], b < p, and [[1, 0], [0, p]]); p != ell prime.
std::vector<uint64_t> ApplyTpOnH0(uint32_t n_sym, uint64_t ell, uint32_t e, uint64_t p, const std::vector<uint64_t>& v);

// ---- cor:ordprecision -------------------------------------------------------
// h = rank H^1(Gamma; V) (x) Z_ell = 2 d_k - 1 (n even, k = n + 2), 0 (n odd): known
// before the run (cor:ordprecision, v1.12.0-pre.2 R4).
uint32_t FreeRankH1(uint32_t n_sym);
// K = e' + e_max.  ell >= 5, n >= 1, e' >= 1; ell in {2, 3} refused by name.
uint32_t CertifiedWorkingPrecision(uint32_t n_sym, uint64_t ell, uint32_t e_target);
// The refusals of CertifiedWorkingPrecision alone (no pre-pass run): for a caller
// that runs TorsionExponentPrepass itself and forms K = e' + e_max (slot-12 WP-11, A10).
void RequirePrecisionUse(uint32_t n_sym, uint64_t ell, uint32_t e_target);

// ---- lem:ordexponent --------------------------------------------------------
struct OrdinaryExponent {
    uint64_t ell = 0;
    uint32_t g = 0, K = 0, c = 0;       // N = ell^{K-1+c} * prod_{f=1..g} (ell^f - 1)
    uint32_t ell_power() const { return K - 1 + c; }
    double log2() const;                // an upper bound: (K-1+c + g(g+1)/2) log2 ell (rmk:ordexponentuse(3))
    bool fits_u128() const;
    u128 value() const;                 // ENGINE_CHECKs fits_u128()
};
OrdinaryExponent OrdinaryExponentOf(uint32_t g, uint32_t K, uint64_t ell);

// ---- matrices over (+)_i Z/ell^{a_i} ----------------------------------------
struct RingMat {
    uint64_t ell = 0;
    std::vector<uint32_t> row_exp;      // a_i per row (== per column: a square endomorphism)
    std::vector<uint64_t> a;            // row-major, a[i*n + j] in [0, ell^{a_i})
    uint32_t n() const { return static_cast<uint32_t>(row_exp.size()); }
    uint64_t at(uint32_t i, uint32_t j) const { return a[static_cast<size_t>(i) * n() + j]; }
    bool operator==(const RingMat& o) const { return ell == o.ell && row_exp == o.row_exp && a == o.a; }
    bool operator!=(const RingMat& o) const { return !(*this == o); }
};
RingMat Identity(uint64_t ell, const std::vector<uint32_t>& row_exp);
RingMat Canonical(RingMat M);                          // reduce row i mod ell^{a_i}
RingMat Mul(const RingMat& A, const RingMat& B);
RingMat Sub(const RingMat& A, const RingMat& B);
RingMat PowU128(const RingMat& A, u128 e);             // binary powering, A^0 = 1
RingMat Pow(const RingMat& A, const OrdinaryExponent& N);   // A^N by N's factors
RingMat Reduce(const RingMat& A, uint32_t K);          // row exponents min(a_i, K), entries reduced
bool IsZero(const RingMat& A);
uint32_t RankModEll(const RingMat& A);                 // rank over F_ell of A mod ell
// U^{N_ord(g, K)}, g = generator count (rows), K = max row exponent (lem:ordexponent(3)).
RingMat OrdinaryProjector(const RingMat& U);

} // namespace ordinary
} // namespace hecke

#endif // HECKE_ORDINARY_PART_H
