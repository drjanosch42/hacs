// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

#pragma once
// CyclicCubic.h -- the cyclic cubic field K of prime conductor ell (ell = 1 mod 3), its class number by
// the cyclotomic-unit index, and membership of ell in the set Lambda-bar of arXiv:2610.17865 (hecke-extensions
// slot 23, package C3; docs/implementation/slot23/arxiv_2610_17865_assessment.md sec. 5, C3). STL only.
//
// Model. g a primitive root mod ell, H = <g^3> the index-3 subgroup of (Z/ell)^*, C_i = g^i H, and the Gaussian
// periods eta_i = sum_{a in C_i} zeta^a (i = 0, 1, 2). {eta_0, eta_1, eta_2} is a normal integral basis of O_K
// (ell is tamely ramified), sigma: zeta -> zeta^g acts by eta_i -> eta_{i+1}, and 1 = -(eta_0 + eta_1 + eta_2).
// The structure constants eta_i eta_j = sum_k T_ij^k eta_k are exact integer counts. The minimal polynomial of
// eta_0 is x^3 + x^2 - (ell-1)/3 x - (ell (a+3) - 1)/27 with 4 ell = a^2 + 27 b^2, a = 1 mod 3; its discriminant
// is ell^2 b^2 and d_K = det Tr(eta_i eta_j) = ell^2 (checked exactly, not assumed).
//
// Class number (the certified route). C = <-1, v, sigma v> with v = N_{Q(zeta)^+/K}(xi_g),
// xi_g = zeta^{(1-g)/2}(1 - zeta^g)/(1 - zeta); v^2 = u := sigma(xi)/xi with xi = N_{Q(zeta)/K}(1 - zeta) (so
// <u, sigma u> alone has index 4h; the start_power = 2 control reproduces that). [E_K : C] = h_K (classically Hasse
// 1952) by the Dedekind determinant: v has log vector l_j = (Lc_{j+1} - Lc_j)/2 with Lc_j = sum_{a in C_j} log|1 - zeta^a|
// and sum_j l_j = 0, so R(C) = (1/3) prod_{chi != 1} |sum_j chi(g^j) l_j| = (1/3) prod_{chi != 1} (|1 - chi(g)|/2)
// |sum_a chi(a) log|1 - zeta^a|| = (1/4) |sum_a chi(a) log|1 - zeta^a||^2 (|1 - omega|^2 = 3) = ell |L(1,chi)|^2 / 4 = hR
// (Washington GTM 83, Lemma 5.26, and the class number formula). The index is found by saturation: for every
// prime q up to the index bound R_cur / R_low, with Cusick's unconditional lower bound
// R_K >= (1/16) log^2(d_K / 4) for totally real cubic fields (Cusick 1984), every class of C/C^q is tested for
// being a q-th power in O_K. A found root is an exact algebraic integer, checked exactly (r^q = +-c in BigInt
// arithmetic); a rejection rests on one numerical step (the traces Tr(r eta_k), integers if r is integral, are
// computed in MpReal at P bits with >= 80 bits over their magnitude -- try_root NF_CHECKs this against P -- and
// must lie within 2^-20 of an integer). MpReal has no interval arithmetic, so the rejections are certified up to
// its documented rounding contract.
//
// Independent check (of the identity above, not of the saturation): hR = ell |L(1,chi)|^2 / 4 with chi the cubic
// character mod ell and L(1,chi) = -(1/ell) sum_a chi(a) psi(a/ell) (digamma, long double; no logarithm of a sine),
// against R(C) from the unit route (rel_err). h_an = hR / R is only a consistency check: R = R(C) / [index found],
// so h_an equals the computed index times (1 + rel_err) whether or not the saturation is complete; completeness
// rests on Cusick's bound and the trace rejections. Structural checks: 3 does not divide h (genus theory,
// one ramified prime), every prime p = 2 mod 3 divides h to an even power, so v_2(h) is even, and the 2-rank of
// Cl(K) is even (Cl[2] is an F_4-vector space since Cl^G = 1).
//
// Lambda-bar (arXiv:2610.17865, item (2) after eq. (1)): ell = 1 mod 3 and the 2-rank of Cl(K) is >= 1. Since
// the 2-rank is even, this is 2 | h <=> 4 | h.
#include <array>
#include <cstdint>
#include <string>
#include <vector>

#include "BigInt.h"

namespace nf {

// An element of O_K in the normal basis (x_0 eta_0 + x_1 eta_1 + x_2 eta_2).
using CubicElt = std::array<BigInt, 3>;

struct CyclicCubicField {
    uint32_t ell = 0;
    uint32_t g = 0;                 // the least primitive root mod ell
    std::vector<uint8_t> coset;     // coset[x] = i with x in C_i, x = 1..ell-1 (coset[0] unused)
    int64_t T[3][3][3] = {};        // eta_i eta_j = sum_k T[i][j][k] eta_k
    int64_t a = 0, b = 0;           // 4 ell = a^2 + 27 b^2, a = 1 mod 3, b > 0
    int64_t e2 = 0, e3 = 0;         // elementary symmetric functions of the periods (e1 = -1), exact
    int64_t poly[4] = {};           // x^3 + poly[2] x^2 + poly[1] x + poly[0] (poly[3] = 1), from e1, e2, e3
    BigInt poly_disc;               // discriminant of the period polynomial
    BigInt basis_disc;              // det(Tr(eta_i eta_j)) = d_K
    int64_t G[3][3] = {};           // the trace form Tr(eta_i eta_j)
};

// Builds the field; NF_CHECK(ell prime, ell = 1 mod 3, 7 <= ell < 2^20).
CyclicCubicField cyclic_cubic_field(uint32_t ell);

CubicElt cubic_mul(const CyclicCubicField& K, const CubicElt& x, const CubicElt& y);
CubicElt cubic_sigma(const CubicElt& x);  // eta_i -> eta_{i+1}
CubicElt cubic_one();
CubicElt cubic_pow(const CyclicCubicField& K, CubicElt x, uint64_t e);
BigInt cubic_norm(const CyclicCubicField& K, const CubicElt& x);  // exact
bool cubic_is_rational(const CubicElt& x, BigInt* value);         // x = value * 1 ?

struct CubicClassResult {
    uint32_t ell = 0;
    int64_t a = 0, b = 0;
    bool field_ok = false;          // poly = expected form, poly_disc = ell^2 b^2, basis_disc = ell^2
    bool units_ok = false;          // xi exact with N(xi) = ell, u = sigma(xi)/xi exact, N(u) = 1, u sigma(u) sigma^2(u) = 1
    uint64_t h = 0;                 // [E : C] = h_K (certified, see the header)
    unsigned v2 = 0;                // v_2(h)
    int rank2 = -1;                 // 2-rank of Cl(K) when determined by v2 (0 if v2 = 0, 2 if v2 = 2), else -1
    bool in_lambda_bar = false;     // 2 | h
    uint64_t sat_bound = 0;         // the largest prime q the saturation had to test
    uint32_t root_tests = 0;        // classes tested
    std::vector<uint32_t> roots;    // the primes q of the found q-th roots, in order (product = h)
    double RC = 0, R = 0;           // regulators of C and of E_K
    double hR_analytic = 0;         // ell |L(1,chi)|^2 / 4 (digamma)
    double h_analytic = 0;          // hR_analytic / R (a consistency check only, see the header)
    double rel_err = 0;             // |hR_analytic - RC| / RC (the independent check)
    double R_low = 0;               // Cusick's lower bound
    double min_reject_log2 = 0;     // diagnostic only: log2 distance to Z of the nearest rejected non-root (says nothing about soundness)
    unsigned prec = 0;              // MpReal working precision (bits)
    std::string root_certificate;   // q:coefficients of the first found root (empty if h = 1)
};

// The class number and the Lambda-bar verdict of the cyclic cubic field of conductor ell. extra_prec adds bits to
// the working precision (the determinism check runs it at two precisions). start_power = k > 1 is a control of the
// saturation: it starts from <-1, v^k, sigma(v)^k> (index k^2 h), so the result's h field is then k^2 h_K and
// v2 / rank2 / in_lambda_bar refer to that index, not to Cl(K). 1 <= k <= 4.
CubicClassResult cyclic_cubic_class_number(uint32_t ell, unsigned extra_prec = 0, unsigned start_power = 1);

// L(1, chi) for chi = the cubic character mod ell with chi(g) = e^{2 pi i/3}: hR = ell |L|^2 / 4. Long double.
double cubic_hR_analytic(const CyclicCubicField& K);

}  // namespace nf
