// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

#pragma once
#ifndef HECKE_EXT_LVALUE_MODULAR_SYMBOL_MEASURE_H
#define HECKE_EXT_LVALUE_MODULAR_SYMBOL_MEASURE_H

// ===========================================================================
// (hecke-extensions slot 20, WP-MEAS) THE MEASURE-VALUED EVALUATION OF A LINEAR ROUTE:
// the ell-stabilised Mazur-Tate-Teitelbaum measure of an ordinary eigensymbol on [C]'s
// 7-vertex model.  [E] 1.20.0-pre.1 sec:lvmeasure: def:lvmeasure (a measure is a family of
// linear functionals, one per ball; owner ruling O5 of 2026-10-04: NO fourth kind),
// def:lvmtt, prop:lvmtt, cl:lvmeascong, rmk:lvmeasplus, chk:lvmeas.  Spec
// docs/implementation/slot20/WP-MEAS_spec.md; theory and the numbers of record
// docs/implementation/slot20/R20-1_measures.md (Python oracle r201/measlib.py).
//
// THE MEASURE (def:lvmtt).  For P = Phi({0,oo}) of a T_ell-eigenclass with a_ell an ell-unit,
// alpha the unit root of X^2 - a_ell X + ell^{k-1}, gamma_{a,m} = [[ell^m, -a], [0, 1]]:
//     c(a, m)  = < rho_n(gamma_{a,m}) Phi({a/ell^m, oo}), u_n^vee > = - sum_i P(q_i, q'_i)   (m >= 1)
//                (bottom rows (q_i, q'_i) of Manin's matrices of a/ell^m, P(c,d) = sum_t P_t c^{n-t} d^t),
//     c(a, 0) = c(a, -1) = P_n,
//     mu(a + ell^m Z_ell) = alpha^{-m} c(a, m) - alpha^{-m-1} ell^{k-2} c(a, m-1)              (m >= 0).
// The factor ell^{k-2} is part of the definition (without it the family is not a distribution:
// the negative control MeasureRequest::ell_n_factor = false).
//
// THE KUBOTA-LEOPOLDT SIDE (cl:lvmeascong).  G_b (b != 0 mod ell-1, [U] lem:u-twistvalues) as an
// element of Z/ell^R[C_{ell^{M-1}}], index i <-> [u^i], u = 1 + ell, by two routes:
//   KlGroupRingBernoulli  the regularised Bernoulli measure E_{1,c} (Washington Sec. 12.1, recalled),
//                         G_b = -N_b / (1 - omega^b(c)[<c>]); sign calibrated by r201/kl_check.py;
//   KlGroupRingLayer      NumberField's Coates-Wiles layer (CwLayer, Bernoulli-free): G_b = -(ell-1) nu_b.
//
// STL + NumberField/CwLayer.h (in the .cpp only) + the slot-19 L-value headers.  Pure host arithmetic;
// writes nothing; no golden object; no record.  Arithmetic mod ell^R < 2^62 (the hot loop uses a Barrett
// reduction for ell^R < 2^32); the c-tables are parallel in `a` (std::thread, ENGINE_THREADS respected).
// ===========================================================================

#include "BirchModularSymbol.h"
#include "LValueRoute.h"

#include <cstdint>
#include <map>
#include <string>
#include <vector>

namespace hecke_ext {
namespace lvalue {

// (slot 20, WP-MEAS; [E] def:lvmeasure, rmk:lvshapes (M)) A measure is a family of linear functionals:
// x |-> mu_x(a + ell^M Z_ell), a in Z/ell^M.  Only kind() == Linear routes answer; the others refuse_kind("measure").
struct MeasureRequest {
    uint32_t level = 2;                 // M >= 1: balls a + ell^M Z_ell; branch elements live on Gamma / Gamma^{ell^{M-1}}
    bool ell_n_factor = true;           // false = the NEGATIVE CONTROL (the plan's formula without ell^{k-2}); never a production value
    int64_t alpha_override = -1;        // >= 0: use this root (NEGATIVE CONTROL "wrong root"); -1: the unit root
};

struct IwasawaInvariants {
    uint32_t branch = 0;
    bool vanishes = false;              // the whole branch element is 0 mod ell^R (wrong parity)
    uint32_t mu = 0;                    // min valuation of the T^t-coefficients t < min(ell, ell^{M-1}) (capped at R)
    uint32_t lambda = 0;                // first t with valuation mu
    bool readable = true;               // false: mu >= min(R, M-1) or lambda >= ell^{M-1} (not determined at this level)
};

struct ModularSymbolMeasure {
    uint64_t ell = 0; uint32_t R = 0; uint32_t M = 0; uint32_t n = 0;   // values mod ell^R
    uint64_t q = 0;                     // ell^R
    uint64_t a_ell = 0, alpha = 0;
    bool ell_n_factor = true;
    int geometric_sign = 0;             // +1 / -1 / 0 (mixed)
    std::vector<std::vector<uint64_t>> c;    // c[m+1][a] = c(a, m), m = -1 .. M (c[0] = {P_n})
    std::vector<std::vector<uint64_t>> mu;   // mu[m][a] = mu(a + ell^m), m = 0 .. M, a = 0 .. ell^m - 1

    uint64_t ball(uint64_t a, uint32_t m) const;                 // mu(a + ell^m Z_ell), m <= M
    std::vector<uint64_t> branch(uint32_t a) const;              // g_i = sum_delta omega^a(delta) mu(delta u^i + ell^M), i < ell^{M-1}
    uint64_t branch_value(uint32_t a, uint32_t j) const;         // sum_i g_i u^{ij} = int omega^{a-j}(x) x^j dmu  (mod ell^{min(M,R)})
    uint64_t moment(uint32_t j, uint32_t i) const;               // int omega^{-i}(x) x^j dmu by the level-M Riemann sum (mod ell^{min(M,R)})
    // moments for all j = 0..n and each listed character index i (one pass over the balls): out[ii][j].
    std::vector<std::vector<uint64_t>> moments(const std::vector<uint32_t>& chars) const;
    IwasawaInvariants invariants(uint32_t a) const;
    uint64_t fnv_units(uint32_t level = 2) const;                // FNV-1a-64 (Fnv.h) of mu(a + ell^level), units a ascending, 8 LE bytes each
    uint64_t unit_total(uint32_t level) const;                   // mu(Z_ell^x) from the level-`level` unit balls
    NormalisationRecord normalisation() const;                   // the interpolation formula, R20-1 Sec. 2.4 (printed with every value)
};

namespace birch {

// Barrett-reduced arithmetic mod q (q < 2^32 uses the fast path; else unsigned __int128 %).
uint64_t MulModFast(uint64_t a, uint64_t b, uint64_t q);

// The unit root alpha of X^2 - a_ell X + ell^{k-1} mod ell^R (a_ell a unit): iterate x <- a - ell^{k-1}/x; checked root.
uint64_t UnitRoot(const Ring& R, uint64_t a_ell, uint32_t k);
// The measure tables of P to level M (threads: 0 = ENGINE_THREADS or the hardware, capped at 16).
ModularSymbolMeasure BuildMeasure(const SymRho& rho, const Vec& P, uint64_t a_ell, uint64_t alpha, uint32_t M, bool ell_n_factor,
                                  unsigned threads = 0);
// sum_b c(a + b ell^m, m+1) == a_ell c(a,m) - ell^{k-2} c(a,m-1), a < ell^m: the number of failing a.
uint32_t ThreeTermFailures(const ModularSymbolMeasure&, uint32_t m);
// sum_b mu(a + b ell^m, m+1) == mu(a, m), a < ell^m: the number of failing a.
uint32_t DistributionFailures(const ModularSymbolMeasure&, uint32_t m);

// The symbol-side cell: W^{+-} = ker(1+S) cap ker(1+U+U^2) on the even / odd coordinates (Smith over Z/ell^R), T_p on W,
// the eigen-system of `sep` (sum c_p T_p): the root of charpoly(sep | W^-) congruent to sum c_p e_p (Eisenstein), or at a
// control (`control` = true, no Eisenstein root) the first root mod ell; simple mod ell (else REFUSED with the multiplicity),
// Hensel-lifted; r^- (first unit coordinate 1), r^+ (the same eigenvalue on W^+; r^+_0 = 1), a_p for p in {2,3,5,7,ell}.
struct SymbolCell {
    uint32_t n = 0;
    uint64_t ell = 0;
    uint32_t R = 0;
    Ring ring;
    std::vector<Vec> Wminus, Wplus;
    std::vector<uint64_t> charpoly_roots;    // roots mod ell of charpoly(sep | W^-)
    uint64_t target = 0;                     // the root used (mod ell)
    bool eisenstein_congruent = false;       // the Eisenstein root sum c_p e_p occurs mod ell
    uint64_t lam = 0;                        // its Hensel lift
    Vec r_minus, r_plus;
    bool have_r_plus = false;
    std::vector<uint32_t> wplus_pivots;      // Smith pivot valuations of (sep - lam) on W^+
    std::map<uint64_t, uint64_t> a_p;
    uint32_t target_multiplicity = 0;
    bool simple = false;
    std::string refusal;
    double seconds = 0;
};
SymbolCell RunSymbolCell(uint32_t n, uint64_t ell, uint32_t R, const std::map<uint64_t, uint64_t>& sep, bool control);

// The Eisenstein symbol P^Eis_{n-j} = C(n,j) (B_{j+1}/(j+1)) (B_{n-j+1}/(n-j+1)), odd j (exact Bernoulli numbers; C(n,j) exact).
// min_valuation: the least ell-adic valuation of a non-zero coordinate (P^Eis is returned mod ell^R; refused if < 0).
Vec EisensteinSymbol(uint32_t n, const Ring& R, int32_t* min_valuation = nullptr);
uint32_t WResidualValuation(const SymRho& rho, const Vec& P);           // P in W(Z/ell^e) iff result >= e

// An ell-adic number read as (valuation, unit part mod ell^R); `zero` = exactly 0 at the working precision.
struct LadicReading {
    int32_t v = 0;
    uint64_t unit = 0;
    bool zero = false;
};
// B_r / r exactly (tangent numbers, one table per kmax), r >= 2 even; v may be negative ((ell-1) | r).
class BernoulliOverKTable {
public:
    BernoulliOverKTable(uint64_t ell, uint32_t R, uint32_t kmax);
    LadicReading operator()(uint32_t r) const;                        // r even, 2 <= r <= kmax
    uint64_t residue(uint32_t r) const;                               // B_r / r mod ell^R (requires v >= 0)
private:
    uint64_t ell_ = 0;
    uint32_t R_ = 0;
    std::vector<LadicReading> tab_;
};
// B_{r, omega^i} / r (i != 0 mod ell - 1), the definition B_{r,chi} = sum_s C(r,s) B_s ell^{s-1} S_{r-s}, S_q = sum_a chi(a) a^q,
// at working precision ell^Q (ell^Q < 2^62); v may be -1 (the pole: i + r == 0 mod ell - 1).  R20-1's TwistedBernoulli in integers.
class TwistedBernoulliTable {
public:
    TwistedBernoulliTable(uint64_t ell, uint32_t R, uint32_t rmax);
    LadicReading operator()(uint32_t r, uint32_t i) const;            // cached
private:
    uint64_t ell_ = 0, NN_ = 0;
    uint32_t R_ = 0, Q_ = 0, rmax_ = 0;
    std::vector<uint64_t> Ls_;                                        // ell^s B_s mod ell^Q
    mutable std::map<uint32_t, std::vector<uint64_t>> S_;             // S_i[q]
    mutable std::map<std::pair<uint32_t, uint32_t>, LadicReading> cache_;
};

// Kubota-Leopoldt G_b (b != 0 mod ell-1) as an element of Z/ell^R[C_{ell^{M-1}}], index i <-> [u^i] <-> (1+T)^i, two routes:
std::vector<uint64_t> KlGroupRingBernoulli(uint32_t b, uint64_t ell, uint32_t M, uint32_t R); // E_{1,c}: G_b = -N_b / (1 - omega^b(c)[<c>])
// NumberField CwLayer at level M: G_b = -(ell-1) nu_b mod ell^M for every listed class b (one lambda_M serves all classes).
std::map<uint32_t, std::vector<uint64_t>> KlGroupRingLayer(const std::vector<uint32_t>& classes, uint64_t ell, uint32_t M, unsigned threads);
std::vector<uint64_t> GrMul(const std::vector<uint64_t>&, const std::vector<uint64_t>&, uint64_t q);
std::vector<uint64_t> GrTw1(const std::vector<uint64_t>&, uint64_t ell, uint64_t q);           // G(u(1+T) - 1):              g_i -> u^i g_i
std::vector<uint64_t> GrTwInv(const std::vector<uint64_t>&, uint64_t ell, uint64_t q, uint32_t k); // G(u^{k-1}(1+T)^{-1} - 1): h_{-i} = u^{(k-1)i} g_i
uint64_t GrEval(const std::vector<uint64_t>& g, uint64_t s, uint64_t ell, uint64_t q);          // sum_i g_i u^{i s}

// The Teichmuller character omega^{-i} mod ell as a DirichletCharacter of conductor ell (chibar = omega^{-i}), for EvAllJ.
DirichletCharacter TeichmullerCharacter(uint32_t i, const Ring& R);

unsigned MeasureThreads();                                            // ENGINE_THREADS (default: hardware), capped at 16

} // namespace birch
} // namespace lvalue
} // namespace hecke_ext

#endif // HECKE_EXT_LVALUE_MODULAR_SYMBOL_MEASURE_H
