// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

#pragma once
#ifndef HECKE_EXT_HECKE_LAMBDA_TWOVAR_H
#define HECKE_EXT_HECKE_LAMBDA_TWOVAR_H

// ===========================================================================
// (hecke-extensions slot 21, P9) THE TWO-VARIABLE MEASURE BY INTERPOLATION.
// Spec: docs/implementation/slot21/P9_spec.md v0.2.  Source: the owner's
// roadmap target (ii), row P9 (the p-adic Hida note).  Inputs: P8's normalised unit-ball masses nu_B(x_k)
// (HeckeLambdaModSym, [U] v0.5.0-pre.2 part 4), [E] prop:lvmtt(5) (branch
// elements), def:lvlfun (the Taylor shift), [U] thm:u-interp (interpolation).
//
// WHAT IS HERE (pure arithmetic, no cohomology):
//  * the branch element g^{(a)}_i = sum_delta omega^a(delta) mu(delta u^i + ell^M)
//    of any family of unit-ball masses, and the Taylor shift c_t = sum_i C(i,t) g_i;
//  * the Lagrange weights of thm:u-interp at an arbitrary point y of the disc
//    x_c + ell Z_ell (an integer weight, the Eisenstein point alpha_a, x_2): the
//    weights lambda_j(y) are integral, and ell lambda_j'(y) is integral (the
//    derivative); predictions are correct mod ell^{min(K', N)};
//  * the zero alpha of a Lambda-function with a simple zero in the disc, from its
//    values at N consecutive weights (Hensel on the Lagrange interpolant);
//  * 11a: weight-2 modular symbols on Gamma_0(11) from Manin symbols on P^1(Z/11)
//    (independent of every level-one routine), its MTT measure (a multiplicative
//    prime: mu(a + 11^m) = a_11^{-m} Phi({a/11^m, oo})), and Tate's L-invariant
//    log_11(q_E) / ord_11(q_E) of 11a1 from j = E_4^3 / Delta by series inversion;
//  * the Iwasawa logarithm log_ell on Z_ell^x.
//
// STL + EngineCheck.h + HeckeHidaLambda.h (ZMod, XAt) + BirchModularSymbol.h (Ring,
// Smith).  Writes nothing; no golden object; no digest of record.
// ===========================================================================

#include <cstdint>
#include <functional>
#include <string>
#include <vector>

#include "HeckeHidaLambda.h"
#include "LValue/BirchModularSymbol.h"

namespace hecke_ext {
namespace lambda {
namespace twovar {

// The Teichmueller lift of d (ell !| d) mod ell^e.
uint64_t Teich(uint64_t d, uint64_t ell, uint32_t e);

// The branch element of branch a at level M from unit-ball masses mass(b) = mu(b + ell^M), b < ell^M, ell !| b:
// g_i = sum_{delta in mu_{ell-1}} omega^a(delta) mass(delta u^i mod ell^M), i < ell^{M-1}, mod q = ell^Kp.
std::vector<uint64_t> BranchFromBalls(const std::function<uint64_t(uint64_t)>& mass, uint64_t ell, uint32_t M, uint32_t a, uint32_t Kp);
// c_t = sum_i C(i, t) g_i mod q (the Taylor shift X = 1 + T), t < g.size().
std::vector<uint64_t> TaylorShift(const std::vector<uint64_t>& g, uint64_t q);

// x_k = (1 + ell)^k - 1 mod ell^e (k >= 0).
u128 XOf(uint64_t ell, uint64_t k, uint32_t e);

// Lagrange weights at y (given mod ell^{ey}, y == x_{weights[0]} mod ell) for N consecutive weights of one class
// (N <= ell; ENGINE_CHECK otherwise): deriv = false: lambda_j(y) mod ell^Kp; deriv = true: ell * lambda_j'(y) mod ell^Kp.
// The result is exact mod ell^{min(Kp, ey - 1)} (ey >= Kp + N - 1 for exact weights; the caller states its own precision).
std::vector<uint64_t> LagrangeAt(uint64_t ell, const std::vector<uint32_t>& weights, uint32_t Kp, u128 y, uint32_t ey, bool deriv);
// sum_j w_j v_j mod ell^Kp.
uint64_t Combine(const std::vector<uint64_t>& w, const std::vector<uint64_t>& v, uint64_t ell, uint32_t Kp);

// The zero alpha in the disc of the class of a Lambda-function G with a SIMPLE zero there (G = eps (X - alpha), eps a unit),
// from G(x_{k_j}) mod ell^P at N consecutive weights: Hensel on the Lagrange interpolant; alpha mod ell^{min(P, N)}.
// residual (out): G~(alpha) mod ell^{min(P, N)} (0 on success).
u128 ZeroInDisc(uint64_t ell, const std::vector<uint32_t>& weights, const std::vector<uint64_t>& values, uint32_t P, uint64_t* residual);

// The Iwasawa logarithm of a unit x mod ell^prec (log(x^{ell-1}) / (ell-1)); log(x) / ell when div_ell (then mod ell^{prec}).
uint64_t LogIwasawa(uint64_t x, uint64_t ell, uint32_t prec, bool div_ell = false);

// ---- 11a: weight-2 modular symbols on Gamma_0(N), N prime (Manin symbols on P^1(Z/N)) ----
struct Gamma0Symbol {
    uint64_t N = 0;
    int sign = 0;               // psi(-c:d) = sign psi(c:d)
    int64_t ap2 = 0;            // psi o T_2 = ap2 psi (Merel's four matrices of determinant 2)
    lvalue::birch::Ring R;      // coefficients Z/ell^R
    std::vector<uint64_t> psi;  // on P^1(Z/N): index c for (c:1), c < N; index N for (1:0)
    uint32_t gens = 0;          // generators of the solution module
    bool free_rank1 = false;    // the solution module is free of rank 1 (then psi is its generator)
    // Phi({oo, a/M}) = sum over the continued-fraction convergents p_j/q_j of a/M of psi(q_j : (-1)^{j-1} q_{j-1}).
    uint64_t AtCusp(int64_t a, int64_t M) const;
};
// The functionals psi : P^1(Z/N) -> Z/ell^R with psi(x) + psi(xS) = 0, psi(x) + psi(xU) + psi(xU^2) = 0, the sign condition and
// the T_2 condition (N odd prime, ell^R < 2^62).
Gamma0Symbol Gamma0Weight2Symbol(uint64_t N, int sign, int64_t ap2, uint64_t ell, uint32_t R);

// The MTT measure of a weight-2 newform at a prime ell || N with a_ell = ap_ell (a unit): mu(a + ell^m) = ap_ell^{-m} Phi({a/ell^m, oo}),
// Phi({r, oo}) = -Phi({oo, r}).  masses[m][a], m = 0 .. M, mod ell^R.
std::vector<std::vector<uint64_t>> MultiplicativeMeasure(const Gamma0Symbol& s, uint64_t ell, uint32_t M, uint64_t ap_ell);

// Tate's L-invariant L = log_ell(q_E) / ord_ell(q_E) of 11a1 (j = -2^12 31^3 / 11^5) at ell = 11, mod 11^prec (Iwasawa branch,
// log 11 = 0): j(q) = E_4^3 / Delta inverted to q(t), t = 1/j, evaluated at t_E.  prec <= 10.
uint64_t TateLInvariant11a(uint32_t prec);

}  // namespace twovar
}  // namespace lambda
}  // namespace hecke_ext

#endif  // HECKE_EXT_HECKE_LAMBDA_TWOVAR_H
