// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

#pragma once
#ifndef HECKE_EXT_HECKE_HIDA_COMPONENTS_H
#define HECKE_EXT_HECKE_HIDA_COMPONENTS_H

// ===========================================================================
// (hecke-extensions slot 15, WP-6 = slot-14 WP-14c in C++) THE COMPONENT
// READINGS OF [U] AT TAME LEVEL 1.  Design: docs/implementation/slot14/
// WP-14c_report.md Sec. 3; plan docs/implementation/slot15/00-slot15-plan.md
// WP-6.  Mathematics: [U] (docs/latex/hida_hecke_algebra.tex) thm:u-sandwich,
// pr:u-U1, prop:u-loewy(4), pr:u-U3, sec:u-test.  The out-of-tree oracles are
// the WP-C14 scripts docs/implementation/slot14/wpc14/{bernoulli_depth,
// eis_depth_join,hida_types}.py.
//
// Two halves, both console-free, pipeline-side only; no record, no golden
// object, no settings key.
//
// (1) A BIGNUM-FREE B_k/k mod ell^K ([IB] OD-3 (c)), for every prime ell >= 5
//     and K >= 1 with ell^K < 2^64, every even k >= 2 with (ell - 1) not
//     dividing k.  Two routes:
//
//     * the DIRECT route (BernoulliTable): beta_m := ell B_m mod ell^Q for every
//       m <= W by the power-sum identity (Faulhaber at N = ell)
//           ell B_m = S_m(ell) - sum_{t >= 1} C(m, t) (ell B_{m-t}) ell^t / (t+1),
//       S_m(ell) = sum_{0 <= r < ell} r^m.  Every scalar is ell-integral
//       (t - v(t+1) >= 1), no division by ell occurs, so beta_m is EXACT mod
//       ell^Q given the beta_j, j < m; a term with t - v(t+1) >= Q vanishes mod
//       ell^Q.  B_w / w mod ell^K = beta_w / ell^{1 + v(w)} / unit(w), with
//       Q = K + 1 + v(w) (the division is ENGINE_CHECKed exact).  Cost
//       O(W (ell + Q^2)) multiplications mod ell^Q (u128 residues; ell^Q may
//       exceed 2^64, and a 128-bit modulus is multiplied by double-and-add).
//
//     * the KUMMER-MAHLER route (BernoulliOverKModPower): for the class
//       k = k0 + (ell - 1) x, phi(x) := -(1 - ell^{k-1}) B_k / k =
//       L_ell(1 - k, omega^{k0}) = g((1 + q)^x - 1) with g in Z_ell[[Y]] and
//       v(q) >= 1 (the Iwasawa function of a non-trivial even character:
//       f(T, omega^{k0}) in Z_ell[[T]] by Washington, GTM 83, 1st ed., p. 122
//       (Thm 7.1 + Prop 7.6); L_ell(s, chi) = f((1+ell)^s - 1) by Thm 7.10,
//       p. 123; the values by Thm 5.11, p. 57 -- verified on the page images,
//       slot-16 WP-8; g(Y) = f(a-1+aY), a = (1+ell)^{1-k0}), so its Mahler coefficients
//       c_i = (Delta^i phi)(0) satisfy v(c_i) >= i and
//           phi(x) == sum_{i < K} c_i binom(x, i)   (mod ell^K)
//       -- the Kummer congruences in their sharp form.  The truncation at i >= K
//       rests on that citation alone; the run-time check v(c_i) >= i is made for
//       i < K only and is a sanity check, not a proof.  The c_i come from the
//       direct route at the K weights k0 + (ell - 1) j, j < K (all < ell K), so
//       B_k / k mod ell^K costs O(ell^2 K) once per (ell, K, class), for ANY
//       k < 2^64.  The validation cross-checks it against the direct route and
//       against a frozen exact-rational (sympy) oracle.
//
//     Work refusal (by name, std::runtime_error "... refused: work ..."): a
//     predicted work above `budget` multiplications (default 2^34); ell^K
//     >= 2^64 or ell^Q >= 2^127 is refused by name as well.
//
// (2) The READINGS on the shipped Z/ell^K module route
//     (ringalg::RunModule, HeckeOperatorOnKT's H_matrix_module):
//     * the ordinary Eisenstein block e_m H^1(Gamma; Sym^n Z/ell^K), m the
//       system a_p = 1 + p^{k-1} (p != ell), a_ell = 1 (k = n + 2); per p != ell
//       the block charpoly chi_p, the Eisenstein root e_p = 1 + p^{k-1}
//       (checked), the cuspidal factor h_p = chi_p / (X - e_p) (Eichler--Shimura:
//       h_p = c_p^2, c_p the charpoly of T_p on the congruent cusp forms), and
//       L_coh := min_p v(c_p(e_p)) -- thm:u-sandwich's upper bound, exact at
//       g = 1 (h_p = (X - c)^2, L = min_p v(e_p - c)), an upper bound at g >= 2
//       (min_p v(h_p(e_p)) / 2, floor), capped at K;
//     * the Bernoulli side: v(B_k/k) (capped at K) and, on an irregular class,
//       k* and the depth law L_pred = 1 + v(k - k*) ([U] pr:u-U1 + rmk:u-zero);
//       law = agree | LAW_FAIL | COH_FAIL | bound_ok | BOUND_FAIL | na | capped,
//       as eis_depth_join.py;
//     * the g = 2 intra-block reading of every non-Eisenstein ordinary rational
//       block with D = 4 (prop:u-loewy(4)): per p the monic square root c_p of
//       chi_p, v(disc c_p), and the kind (ramified | inert | split | capped), as
//       hida_types.py's intra_block.
//
// Hard-wires neither the 7-vertex model (it reads whatever RunModule's model
// is), a trivial centre, nor Bredon semantics at ell in {2, 3} (ell >= 5 only).
// ===========================================================================

#include <cstdint>
#include <string>
#include <vector>

#include "HeckeRingHeckeAlgebra.h"

namespace hecke_ext {
namespace hidacomp {

using u128 = unsigned __int128;

constexpr uint64_t kDefaultBernoulliBudget = uint64_t(1) << 34;

// ---- (1) Bernoulli numbers mod ell^K, bignum-free ------------------------------

// v_ell(x), x != 0.
uint32_t ValU64(uint64_t x, uint64_t ell);

// The direct route: beta_m = ell B_m mod ell^Q for every m <= W (odd m >= 3 are 0).
class BernoulliTable {
public:
    // ell >= 5 prime, Q >= 2, ell^Q < 2^127, W >= 2.  Throws a named refusal
    // above `budget` predicted multiplications.
    BernoulliTable(uint64_t ell, uint32_t Q, uint64_t W, uint64_t budget = kDefaultBernoulliBudget);
    uint64_t ell() const { return ell_; }
    uint32_t Q() const { return Q_; }
    uint64_t W() const { return static_cast<uint64_t>(beta_.size()) - 1; }
    u128 beta(uint64_t m) const;   // ell B_m mod ell^Q
    // B_w / w mod ell^K, w <= W even, (ell - 1) not dividing w, K + 1 + v(w) <= Q.
    uint64_t BOverK(uint64_t w, uint32_t K) const;
    // in MulMod-equivalent units: x128 when ell^Q >= 2^64 (a 127-step double-and-add MulMod).
    static uint64_t PredictedWork(uint64_t ell, uint32_t Q, uint64_t W);
    // the beta_ vector is refused by name above this many bytes, whatever the work budget.
    static constexpr uint64_t kMaxBetaBytes = uint64_t(2) << 30;   // 2 GiB

private:
    uint64_t ell_;
    uint32_t Q_;
    std::vector<u128> beta_;
};

// B_w / w mod ell^K by the direct route alone (a table up to w).
uint64_t BernoulliOverKDirect(uint64_t w, uint64_t ell, uint32_t K, uint64_t budget = kDefaultBernoulliBudget);

// The Mahler coefficients of one class: c_i = (Delta^i phi)(0) mod ell^K, i < K.
struct ClassSeries {
    uint64_t ell = 0;
    uint32_t K = 0;
    uint64_t k0 = 0;                     // 2 <= k0 <= ell - 3, even
    std::vector<uint64_t> c;             // size K
    uint64_t Phi(uint64_t x) const;      // phi(x) mod ell^K
};
ClassSeries MakeClassSeries(uint64_t ell, uint64_t k0, uint32_t K, uint64_t budget = kDefaultBernoulliBudget);

// B_k / k mod ell^K by the Kummer-Mahler route (any even k >= 2, (ell-1) not | k).
uint64_t BernoulliOverKModPower(uint64_t k, uint64_t ell, uint32_t K, uint64_t budget = kDefaultBernoulliBudget);
uint64_t BernoulliOverKFromSeries(const ClassSeries& s, uint64_t k);

// v_ell(B_k / k) capped at `cap` (lower_bound = true at the cap).
struct CappedVal {
    uint32_t v = 0;
    bool lower_bound = false;
};
CappedVal ValBernoulliOverK(uint64_t k, uint64_t ell, uint32_t cap);

// The irregular indices k0 in {2, ..., ell - 3} (ell | B_k0), by the direct route mod ell^2.
std::vector<uint64_t> IrregularIndices(uint64_t ell, uint64_t budget = kDefaultBernoulliBudget);

// The ell-adic zero of an irregular class: phi(x*) == 0, x* mod ell^{prec-1}, by
// digit-wise Hensel on phi mod ell^prec.  lambda_one = at every digit exactly one
// lift (the simple zero; v(c_1) = 1).  k* = k0 + (ell - 1) x* mod ell^{prec-1}.
struct KStar {
    uint64_t ell = 0, k0 = 0;
    uint32_t prec = 0;
    bool irregular = false, lambda_one = false;
    uint64_t x_star = 0, k_star = 0, modulus = 0;   // modulus = ell^{prec-1}
};
KStar ComputeKStar(uint64_t ell, uint64_t k0, uint32_t prec = 4);

// The depth law L_pred(k) = 1 + v(k - k*) (0 on a regular class), capped at prec
// (lower_bound when ell^{prec-1} | k - k*).  (ell - 1) | k: na = true.
struct DepthPred {
    bool na = false;
    uint32_t L = 0;
    bool lower_bound = false;
};
DepthPred PredictedDepth(uint64_t k, uint64_t ell, uint32_t prec = 4);

// ---- (2) the readings on the module route ----------------------------------------

using Poly = std::vector<uint64_t>;   // ascending, monic, mod ell^K

// (quotient, remainder) of monic f by (X - e) mod q.
std::pair<Poly, uint64_t> SyntheticDivide(const Poly& f, uint64_t e, uint64_t q);
// the monic square root r (deg f / 2) of a monic f of even degree mod q (q odd), ok = r^2 == f.
std::pair<Poly, bool> MonicSqrt(const Poly& f, uint64_t q);
// v_ell(x mod q) capped at cap (q = ell^cap).
uint32_t ValCapped(uint64_t x, uint64_t ell, uint32_t cap);

struct EisReading {
    bool present = false;                // the Eisenstein system is a block of H^1(F_ell)
    bool free = false;                   // e_m H^1(Z/ell^K) free over Z/ell^K
    uint32_t D = 0, g = 0;               // D = 1 + 2 g
    bool eis_root = false;               // chi_p(e_p) == 0 mod ell^K at every p != ell read
    bool square_ok = false;              // h_p is a square (c_p^2) at every p
    std::string kind;                    // exact | upper_bound | torsion_block | absent
    uint32_t L = 0;                      // L_coh (min over p), capped at K
    bool capped = false;                 // no p attains L below K
    std::vector<uint64_t> primes;        // p != ell read
    std::vector<uint32_t> per_p;         // v(e_p - c_p) (g = 1) or floor(v(h_p(e_p)) / 2)
    std::vector<Poly> cusp_factor;       // c_p (monic, degree g), per p
};

struct IntraBlock {
    std::vector<uint64_t> system;        // aligned with ComponentCell::primes
    uint32_t D = 0, g = 0;
    bool square_ok = false;
    std::vector<uint32_t> disc_v;        // per p != ell, capped at K
    std::vector<std::string> kinds;      // ramified | inert | split | capped
    std::vector<Poly> cusp_factor;
    std::string verdict;                 // one_component_ramified | one_component_inert |
                                         // two_rational_eigenforms | capped | INCONSISTENT_* | g_ge3_disc_only
    uint32_t level2 = 0;                 // 2 * level (level = min disc_v / 2)
};

struct ComponentCell {
    uint32_t n = 0;
    uint64_t ell = 0;
    uint32_t K = 0;
    uint64_t k = 0;                      // n + 2
    std::vector<uint64_t> primes;        // ascending, ell included
    std::vector<uint32_t> exps;          // H^1(Z/ell^K) exponents
    EisReading eis;
    // the Bernoulli side
    bool class_zero = false;             // (ell - 1) | k
    CappedVal v_bk;                      // v(B_k / k), capped at K
    DepthPred pred;                      // the depth law at prec max(K, 4)
    std::string law;
    std::vector<IntraBlock> intra;       // the non-Eisenstein ordinary rational blocks with D = 4
    double seconds = 0.0;
};

struct ComponentOptions {
    std::vector<uint64_t> primes;        // S; empty = {2, 3, 5, 7, ell}
    ringalg::ModuleRoute route = ringalg::ModuleRoute::Native;
    bool intra = true;                   // read the g = 2 blocks
};

// n >= 2 even, ell >= 5 prime, K >= 2, ell^K < 2^64.
ComponentCell ComputeComponentCell(uint32_t n, uint64_t ell, uint32_t K, const ComponentOptions& opts = {});

// "HIDA_COMPONENT n=.. ell=.. K=.. k=.. D=.. g=.. L_coh=.. kind=.. v_bk=.. L_pred=.. law=.. eis_root=.. per_p=[..]"
std::string ComponentLine(const ComponentCell& c);
// "HIDA_INTRA n=.. ell=.. system=(..) D=.. g=.. disc_v=[..] kinds=[..] verdict=.. level=.."
std::string IntraLine(const ComponentCell& c, const IntraBlock& b);

} // namespace hidacomp
} // namespace hecke_ext

#endif // HECKE_EXT_HECKE_HIDA_COMPONENTS_H
