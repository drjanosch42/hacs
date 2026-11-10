// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

#pragma once
#ifndef HECKE_EXT_LVALUE_COEFFICIENT_RING_MEASURE_H
#define HECKE_EXT_LVALUE_COEFFICIENT_RING_MEASURE_H

// ===========================================================================
// (hecke-extensions slot 21, P3) THE MTT MEASURE OVER A COEFFICIENT RING O = (Z/ell^R)[x]/(F).
// Spec docs/implementation/slot21/P3_spec.md; oracle of record docs/implementation/slot21/p3/ (Python, run_p3.py, which
// generates CoefficientRingFrozenP3.inc).  WP-MEAS (ModularSymbolMeasure.h, [E] def:lvmtt / prop:lvmtt) needs a simple root
// of charpoly(sep | W^-) in F_ell; here the T_2-eigenvalue of a level-1 eigen-system is theta = x in the local monogenic order
//
//     O = (Z/ell^R)[x]/(F),   F the Hensel factor of chi = charpoly(T_2 | W^-) with F == gbar^e (mod ell), gbar irreducible,
//
// (e = 1: O = W(F_q)/ell^R, q = ell^{deg gbar}; deg gbar = 1, e >= 2, F(X + lambda) Eisenstein: Z_ell[x]/(F), totally ramified).
//   * the eigen-symbol P_theta in W^{+-} (x) O: Gauss-Jordan over the local ring O with UNIT pivots (residual multiplicity one on
//     the sign, else refused by name), normalised so that its first coordinate outside the maximal ideal is 1 (index t0);
//   * a_p, alpha (the unit root of X^2 - a_ell X + ell^{k-1} in O; non-ordinary refused by name);
//   * the measure by O-LINEARITY: P = sum_j x^j P_j, c_P = sum_j x^j c_{P_j} (WP-MEAS's BuildMeasure c-tables of the components,
//     unchanged), mu(a + ell^m) = alpha^{-m} c(a,m) - alpha^{-m-1} ell^{k-2} c(a,m-1) in O.
// The ramified route B (RouteBRead) re-reads the same symbol through WP-LV3's m-block reader (MBlockReader.h): the order
// O_pi = Z_ell[pi]/(pi^e - ell u), its roots rho, the eigen-lines; phi_rho : O -> O_pi, x |-> lambda0 + rho, compares the routes.
//
// STL + the slot-19/20 L-value headers.  Host arithmetic mod ell^R < 2^62 (unsigned __int128 products); writes nothing; no golden
// object; no record; no env.  ENGINE_CHECK for invariants; refusals are std::invalid_argument("<reason>") (never a silent value).
// ===========================================================================

#include "BirchModularSymbol.h"
#include "ModularSymbolMeasure.h"

#include <cstdint>
#include <map>
#include <string>
#include <vector>

namespace hecke_ext {
namespace lvalue {
namespace oring {

using birch::Vec;

// ---- O = (Z/ell^R)[x]/(F), F monic (low -> high) of degree d >= 1 ------------------------------------------------------
class ORing {
public:
    using Elt = std::vector<uint64_t>;   // the coefficients of 1, x, ..., x^{d-1}
    ORing() = default;
    ORing(uint64_t ell, uint32_t R, const Vec& F);
    const birch::Ring& base() const { return Z_; }
    uint64_t ell() const { return Z_.ell; }
    uint32_t R() const { return Z_.m; }
    uint64_t q() const { return Z_.q; }
    uint32_t d() const { return d_; }
    const Vec& F() const { return F_; }
    Elt zero() const { return Elt(d_, 0); }
    Elt one() const { return scalar(1); }
    Elt scalar(uint64_t a) const;
    Elt x() const;
    Elt add(const Elt& a, const Elt& b) const;
    Elt sub(const Elt& a, const Elt& b) const;
    Elt neg(const Elt& a) const;
    Elt mul(const Elt& a, const Elt& b) const;
    Elt smul(uint64_t s, const Elt& a) const;
    Elt pow(const Elt& a, uint64_t e) const;
    Elt eval(const Vec& P, const Elt& z) const;     // P over Z/ell^R (low -> high) at z
    bool is_zero(const Elt& a) const;
    bool unit(const Elt& a) const;                  // the multiplication matrix is invertible mod ell (O local: a not in m)
    Elt inv(const Elt& a) const;                    // ENGINE_CHECKs a unit
    uint64_t trace(const Elt& a) const;
    uint64_t norm(const Elt& a) const;
    Elt reduce_to(const ORing& target, const Elt& a) const;   // coefficientwise mod ell^{target.R()} (same F mod ell^{target.R()})
    std::string str(const Elt& a) const;            // "[c0,c1,...]"
    std::string name() const;                       // "Z_37[x]/(x^2 + 29x + 8)"

private:
    birch::Ring Z_;
    Vec F_;
    uint32_t d_ = 0;
    birch::Mat mulmat(const Elt& a) const;          // column j = a x^j
};

// ---- polynomials -------------------------------------------------------------------------------------------------------
// over F_ell (low -> high, trimmed): Rabin's irreducibility test; the multiplicity of g in f.
bool IrreducibleModEll(const Vec& g, uint64_t ell);
uint32_t MultiplicityModEll(const Vec& f, const Vec& g, uint64_t ell);

// ---- the eigen-symbol over O ---------------------------------------------------------------------------------------------
// y != 0 mod m with (A - theta) y = 0 over O (A r x r over Z/ell^R, column convention), when A - theta has residual rank r - 1;
// refuses "residual eigenspace of rank >= 2" otherwise.
std::vector<ORing::Elt> KernelVectorLocal(const ORing& O, const birch::Mat& A, const ORing::Elt& theta);
std::vector<ORing::Elt> NormaliseFirstUnit(const ORing& O, std::vector<ORing::Elt> P, uint32_t* t0);
std::vector<ORing::Elt> HeckeTpO(const birch::SymRho& rho, const ORing& O, const std::vector<ORing::Elt>& P, uint64_t p);
ORing::Elt UnitRootO(const ORing& O, const ORing::Elt& a_ell, uint32_t k);
// sigma(x) for the Galois check: unramified (gbar of degree d, e = 1): the root of F congruent to x^ell; ramified quadratic: -F_1 - x.
ORing::Elt GaloisGenerator(const ORing& O, uint32_t e);
std::vector<ORing::Elt> ApplySigma(const ORing& O, const ORing::Elt& sx, const std::vector<ORing::Elt>& v);

struct OSymbolCell {
    uint32_t n = 0;
    uint64_t ell = 0;
    uint32_t R = 0;
    int sign = -1;
    uint32_t e = 1;
    Vec gbar, F, chi_T2, chi_T3;        // chi_T*: charpoly(T_* | W^-) over Z/ell^R
    uint32_t dimW = 0;
    ORing O;
    ORing::Elt theta;
    std::vector<ORing::Elt> P;          // length n + 1
    uint32_t t0 = 0;
    std::map<uint64_t, ORing::Elt> a_p;
    bool eigen_all = false;             // T_p P == a_p P on every coordinate, every listed p
    bool ordinary = false;
    ORing::Elt alpha;                   // when ordinary
    std::string refusal;                // non-empty: refused by name (eigen-symbol not formed)
    double seconds = 0;
};
// gbar irreducible mod ell, e its multiplicity in charpoly(T_2 | W^-); theta_override: use this root of F instead of x (Galois).
OSymbolCell RunOSymbolCell(uint32_t n, uint64_t ell, uint32_t R, const Vec& gbar, uint32_t e, int sign,
                           const ORing::Elt* theta_override = nullptr);
// The same symbol from a given ring and eigenvalue (the base-change control: any O with theta a T_2-eigenvalue on W^{sign}).
OSymbolCell RunOSymbolCellIn(uint32_t n, const ORing& O, const ORing::Elt& theta, int sign);

// ---- the measure over O ----------------------------------------------------------------------------------------------------
struct OMeasure {
    ORing O;
    uint32_t n = 0, M = 0;
    bool ell_n_factor = true;
    ORing::Elt a_ell, alpha;
    std::vector<std::vector<ORing::Elt>> c;    // c[m+1][a] = c(a, m), m = -1 .. M
    std::vector<std::vector<ORing::Elt>> mu;   // mu[m][a], m = 0 .. M
    uint64_t fnv_units(uint32_t level) const;          // the d coefficients of each unit ball, units ascending (d = 1: WP-MEAS's stream)
    uint64_t fnv_trace_norm(uint32_t level) const;     // (Tr, N) of each unit ball: presentation-free
    ORing::Elt unit_total(uint32_t level) const;
};
OMeasure BuildOMeasure(const birch::SymRho& rho, const ORing& O, const std::vector<ORing::Elt>& P, const ORing::Elt& a_ell,
                       const ORing::Elt& alpha, uint32_t M, bool ell_n_factor = true);
uint32_t ThreeTermFailuresO(const OMeasure& me, uint32_t m);
uint32_t DistributionFailuresO(const OMeasure& me, uint32_t m);
// (-1)^j C(n,j) int omega^{-i} x^j dmu == E_{chi,j} ev_{chi,j}(P) mod ell^{min(M,R)}, all j, the listed i: (failures, total).
std::pair<uint32_t, uint32_t> MomentFailuresO(const birch::SymRho& rho, const OMeasure& me, const std::vector<ORing::Elt>& P,
                                              const std::vector<uint32_t>& chars);

// ---- route B: the m-block reader (WP-LV3) on T_2 | W^{sign} ---------------------------------------------------------------
struct RouteBRoot {
    ORing Opi;                          // (Z/ell^R)[pi]/(pi^e - ell u)
    ORing::Elt rho;                     // the root of the block charpoly (an eigenvalue of T_2 - lambda0), reduced to ell^R
    std::vector<ORing::Elt> P;          // the eigen-symbol over O_pi, normalised (first unit coordinate 1), length n + 1
    uint32_t t0 = 0;
    uint32_t eigen_check = 0;           // v_pi(t g - rho g) from EigenLineAt (at the reader's precision)
};
struct RouteBReading {
    bool complete = false;
    std::string refusal;
    uint32_t orders = 0, roots = 0, e = 0, block_rank = 0;
    std::vector<RouteBRoot> found;
};
RouteBReading RouteBRead(uint32_t n, uint64_t ell, uint32_t R, uint64_t lambda0, int sign);
// phi_rho : O -> O_pi, x |-> lambda0 + rho (checks F(lambda0 + rho) == 0; ENGINE_CHECK).
ORing::Elt PhiRho(const ORing& O, const RouteBRoot& B, uint64_t lambda0, const ORing::Elt& a);

} // namespace oring
} // namespace lvalue
} // namespace hecke_ext

#endif // HECKE_EXT_LVALUE_COEFFICIENT_RING_MEASURE_H
