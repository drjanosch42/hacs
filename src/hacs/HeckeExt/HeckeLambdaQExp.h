// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

#pragma once
#ifndef HECKE_EXT_HECKE_LAMBDA_QEXP_H
#define HECKE_EXT_HECKE_LAMBDA_QEXP_H

// ===========================================================================
// (hecke-extensions slot 21, P7) LAMBDA-ADIC q-EXPANSIONS, TRUNCATED.
// The q-expansion  sum_{n <= B} a_n(X) q^n  mod (ell^K, (X - x_c)^m)  of every
// component of the cuspidal ordinary Hida algebra h^0(a) of tame level 1, read
// as the formal q-expansion with coefficients in the component
// Lambda_{K,m}(x_c)[Y]/(F) (Y = T_{p0}; rank 1 = residue degree 1 = an honest
// Lambda-adic eigenform), plus the Lambda-adic ordinary Eisenstein series of
// the class (a != 0 mod ell - 1).  Spec: the owner's
// p-adic Hida note (roadmap target (ii), P7); plan
// docs/implementation/slot21/00-slot21-plan.md.  Binding text: [U]
// docs/latex/hida_hecke_algebra.tex 0.4.0 part 3 (sec:u-lambda): def:u-trunc,
// thm:u-interp (precision ENFORCED, through InterpolateTruncation),
// prop:u-lambdaprod(4) (the monogenic presentation), thm:u-control (LIT, Hida:
// control for k >= 2), lem:u-twistvalues (G_a), lem:u-lambdaone.
//
// HOW.  The input is WP-HL's q-expansion arm (ComputeFormsWeight: the ordinary
// T_p and U on e S_k(Z/ell^W)) at N >= K + m - 1 consecutive weights of the
// class.  At each weight the residual characteristic polynomial of T_{p0} on
// e S_k is factored over F_ell (Cantor-Zassenhaus); p0 is the least prime
// <= B, != ell, for which it is squarefree at every weight (etale, monogenic:
// [U] (U11)); else REFUSED.  Component g = the generalised g(T_{p0})-part, the
// presentation F = charpoly(T_{p0} | V_g) and P_p(Y), P_U(Y) as polynomials in
// T_{p0} at every weight, interpolated by thm:u-interp (InterpolateTruncation).
// The Lambda-adic weight character eps_p(X) (= p^{k-1} at x_k) is interpolated
// the same way; EpsilonBinomial is the independent closed form
// p^{c-1} sum_j binom(s_p, j) u^{-cj} Z^j, <p> = u^{s_p}.  Then the Hecke
// recursions in the ring: a_1 = 1, a_{mn} = a_m a_n ((m, n) = 1),
// a_{p^{r+1}} = a_p a_{p^r} - eps_p a_{p^{r-1}} (p != ell), a_{ell^r} = alpha^r
// (alpha = the Lambda-adic U).  The Eisenstein series: a_p = 1 + eps_p,
// alpha = 1, a_0 = G~/2 (lem:u-twistvalues, tangent-number B_k/k).
//
// Specialisation at an integer k' in the class: Z -> x_{k'} - x_c; the value
// is determined mod ell^{SpecPrecision} = ell^{min(K, m v(x_{k'} - x_c))}
// (= ell^K at k' = c).
//
// Pipeline side only: pure host C++, STL + HeckeHidaLambda (WP-HL) + EngineCheck.h
// + Fnv.h.  Writes nothing, no golden object, no digest of record.  Every
// refusal is an ENGINE_CHECK (std::logic_error).
// ===========================================================================

#include <cstdint>
#include <map>
#include <string>
#include <vector>

#include "HeckeHidaLambda.h"

namespace hecke_ext {
namespace lambda {

// An element of Lambda_{K,m}(x_c)[Y]/(F): r coefficients of Y^0..Y^{r-1}, each
// a polynomial in Z = X - x_c of m coefficients mod ell^K (low -> high).
using QElem = std::vector<Vec>;

struct LambdaComponent {
    std::string kind;             // "cusp" | "cusp_eis" (residually Eisenstein) | "eisenstein" (the series)
    uint32_t rank = 1;            // r = residue degree (etale)
    std::string gen;              // "T<p0>" (cusp) or "-" (the series)
    std::vector<uint64_t> residual;   // the monic factor g of charpoly(T_{p0}) mod ell, low -> high
    std::vector<Vec> F;           // monic presentation: F[0..r] (F[r] = 1)
    std::map<uint64_t, QElem> ap; // p -> P_p(Y), p prime <= B, p != ell
    QElem alpha;                  // the Lambda-adic U
    std::vector<QElem> a;         // a[0..B]; a[0] = 0 except for the Eisenstein series (G~/2)
};

struct LambdaQExpansion {
    uint64_t ell = 0;
    uint32_t a = 0, K = 0, m = 0, centre_k = 0, B = 0, N = 0, W = 0;
    uint64_t p0 = 0;
    std::vector<uint32_t> weights;        // the N interpolation weights (consecutive in the class)
    std::map<uint64_t, Vec> eps;          // p -> eps_p(Z), p prime <= B, p != ell (interpolated)
    bool has_G = false;
    Vec G;                                // G~ = G_a truncated (a != 0 only)
    std::vector<LambdaComponent> comps;   // cuspidal components (sorted by residual), then the series
};

QElem QScalar(const Vec& z, uint32_t r);
QElem QAdd(const QElem& x, const QElem& y, const ZMod& RK);
QElem QSub(const QElem& x, const QElem& y, const ZMod& RK);
QElem QMul(const QElem& x, const QElem& y, const std::vector<Vec>& F, const ZMod& RK, uint32_t m);
Vec ZMulTrunc(const Vec& x, const Vec& y, const ZMod& RK, uint32_t m);

// x_k = (1 + ell)^k - 1 mod ell^e.
u128 XAt(uint64_t ell, uint64_t k, uint32_t e);
// min(K, m v(x_{k'} - x_c)): the precision of a specialisation at k'.
uint32_t SpecPrecision(uint64_t ell, uint32_t K, uint32_t m, uint32_t centre_k, int64_t kprime);
// The Y-coefficients of x at X = x_{k'}, mod ell^prec (prec <= K).
Vec SpecialiseAt(const QElem& x, uint64_t ell, uint32_t centre_k, int64_t kprime, uint32_t prec);

// The independent closed form of eps_p (p != ell prime, class a, k > 0 in the class
// as the centre): p^{c-1} sum_{j<m} binom(s_p, j) u^{-cj} Z^j mod ell^K, u = 1 + ell,
// <p> = p / omega(p) = u^{s_p} (s_p by a digit-by-digit discrete logarithm).  m < ell.
Vec EpsilonBinomial(uint64_t ell, uint64_t p, uint32_t K, uint32_t m, uint32_t centre_k);

// Build the truncation.  `ws`: forms data (ComputeFormsWeight, same ell and W, the
// primes = every prime <= B except ell) at >= K + m - 1 consecutive weights of the
// class a (the first K + m - 1 are used); W >= K + m - 1 (REFUSED otherwise, by
// thm:u-interp(2)).  p0 = 0: the least etale prime (above); else forced.
// with_eisenstein: add the Lambda-adic Eisenstein series (a != 0 mod ell - 1).
LambdaQExpansion BuildLambdaQExpansion(const std::vector<FormsWeight>& ws, uint32_t a, uint32_t K, uint32_t m,
                                       uint32_t centre_k, uint32_t B, uint64_t p0, bool with_eisenstein);

// The record hecke_lambda_qexp/v1, one line per component: "HECKE_LAMBDA_QEXP ell= a= K= m=
// centre_k= N= W= B= p0= component=<i> kind= rank= gen= residual=[..] F=[[..]..] alpha=[[..]..]
// q=[a_0;a_1;..;a_B] digest=0x<FNV-1a-64 of the line before ' digest='>" (each a_n as
// [[Z-coeffs of Y^0],..]).  Not pinned, no golden line.
std::vector<std::string> QExpRecordLines(const LambdaQExpansion& L);


} // namespace lambda
} // namespace hecke_ext

#endif // HECKE_EXT_HECKE_LAMBDA_QEXP_H
