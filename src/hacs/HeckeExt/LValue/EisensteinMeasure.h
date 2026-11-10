// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

#pragma once
#ifndef HECKE_EXT_LVALUE_EISENSTEIN_MEASURE_H
#define HECKE_EXT_LVALUE_EISENSTEIN_MEASURE_H

// ===========================================================================
// (hecke-extensions slot 21, P2) THE EISENSTEIN MEASURE EXACTLY.  Spec docs/implementation/slot21/P2_spec.md;
// Python oracle docs/implementation/slot21/p2/p2lib.py, run_p2.py.  Builds on WP-MEAS ([E] sec:lvmeasure) and calls
// its public API only (ModularSymbolMeasure.h is not edited).
//
//   Ptil = n! p_k^-  (Zagier 1991 eq. (11), p. 453): Ptil_{n-j} = n! beta_j beta_{n-j}, beta_m = B_{m+1}/(m+1)!, j odd in [-1, n+1],
//          on u_t = x^{n-t} y^t, t = -1 .. n+1 (the interior 0 <= t <= n is WP-MEAS's P^Eis).
//   ctil(a, m) = - sum_i Ptil(q_i, q'_i) over the bottom rows of birch::ManinPath(a, ell^m), the singular monomial dropped.
//   E_r(x + ell^m Z_ell) = ell^{m(r-1)} B_r(x / ell^m) / r;  D(a, m) = sum_{y mod ell^m} Bbar_1(a y / ell^m) E_{k-1}(y + ell^m).
//   mu^Eis(a + ell^m Z_ell) = -D(a, m) + ell^{k-2} D(a, m-1)   (m >= 1; alpha = 1).
// Spec Sec. 0: ctil(a,m) = -D(a,m) + (B_k/k) a / ell^m exactly; on units mu^Eis = -E_1 * iota E_{k-1}; its branch series are
// -G_{a+1}(u(1+T)-1) G_{k-1-a}(u^{k-1}(1+T)^{-1}-1), with the polar term Res / (u(1+T)-1) (branch -1), Res / (u^{k-1}(1+T)^{-1}-1)
// (branch k-1), Res = (1-ell)(log_ell(u)/ell)(1-ell^{k-1}) B_k/k.
//
// Two arithmetics: exact rationals reduced modulo a prime p (p > ell^M, p != ell; the identities over Q are checked there), and
// ell-adic values in a scaled representation ell^E x mod ell^K (K with ell^K < 2^62).  STL + the slot-19/20 L-value headers.
// Pure host arithmetic; writes nothing; no golden object; no record.
// ===========================================================================

#include <cstdint>
#include <vector>

namespace hecke_ext {
namespace lvalue {
namespace eis {

// ---- exact rationals modulo a prime p (p < 2^62, p > every denominator's prime factors used here) ----
uint64_t MulModP(uint64_t a, uint64_t b, uint64_t p);
uint64_t PowModP(uint64_t b, uint64_t e, uint64_t p);
uint64_t InvModP(uint64_t a, uint64_t p);                       // a != 0 mod p
uint64_t FromIntModP(int64_t x, uint64_t p);
// B_0 .. B_N mod p (B_1 = -1/2) by the recurrence (p > N + 1).
std::vector<uint64_t> BernoulliModP(uint32_t N, uint64_t p);
// Zagier's Laurent polynomial n! p_k^- mod p: index t + 1 holds the coefficient of u_t, t = -1 .. n+1.
std::vector<uint64_t> ZagierLaurentModP(uint32_t n, uint64_t p, const std::vector<uint64_t>& B);
// Ptil(c, d) mod p (the Laurent monomials evaluated with c^{-1}, d^{-1}); a singular monomial (c = 0 or d = 0 in its denominator) is
// dropped and *singular counts it.  interior_only: only 0 <= t <= n (WP-MEAS's polynomial P^Eis).
uint64_t EvalLaurentModP(const std::vector<uint64_t>& Pt, uint32_t n, int64_t c, int64_t d, uint64_t p, bool interior_only,
                         uint32_t* singular = nullptr);
// ctil(a, m) mod p along birch::ManinPath(a, M), M = ell^m.
uint64_t CuspSymbolModP(const std::vector<uint64_t>& Pt, uint32_t n, uint64_t a, uint64_t M, uint64_t p, bool interior_only);
// (rho(A) Ptil)(x, y) = Ptil(a x + c y, b x + d y) at a point of F_p^2 (A = [[a, b], [c, d]]).
uint64_t EvalLaurentAt(const std::vector<uint64_t>& Pt, uint32_t n, int64_t a, int64_t b, int64_t c, int64_t d, uint64_t x, uint64_t y,
                       uint64_t p, bool interior_only);

// D(a, m) for every a < ell^m, m = 0 .. M, mod p (threads over a).
struct EisModP {
    uint64_t p = 0, ell = 0;
    uint32_t n = 0, k = 0, M = 0;
    std::vector<std::vector<uint64_t>> D;          // D[m][a]
    uint64_t Bk_over_k = 0;                        // B_k / k mod p
    uint64_t mu(uint64_t a, uint32_t m) const;     // mu^Eis(a + ell^m), m >= 1
};
EisModP BuildEisModP(uint32_t n, uint64_t ell, uint32_t M, uint64_t p, unsigned threads);
// -sum_{x unit} E_1(x) E_{k-1}(x a^{-1}) mod p for every unit a < ell^M (index a; non-units 0).
std::vector<uint64_t> ConvolutionModP(uint32_t n, uint64_t ell, uint32_t M, uint64_t p, unsigned threads);

// ---- ell-adic, scaled: value * ell^E mod ell^K ----
struct EisLadic {
    uint64_t ell = 0;
    uint32_t n = 0, k = 0, M = 0, E = 0, K = 0;
    uint64_t NK = 0;                               // ell^K
    std::vector<uint64_t> mu;                      // mu^Eis(a + ell^M) * ell^E mod ell^K for units a (0 at non-units)
    std::vector<uint64_t> e1, ek;                  // E_1(x) ell^M, E_{k-1}(x) ell^{E-M} (units)
    // g_i = sum_delta omega^br(delta) v(delta u^i + ell^M), i < ell^{M-1} (v = mu, or another table of the same layout)
    std::vector<uint64_t> branch(uint32_t br, const std::vector<uint64_t>* v = nullptr) const;
    int32_t vmin(const std::vector<uint64_t>& g) const;   // min over non-zero entries of v_ell - E (99 if all zero)
    uint64_t sub(uint64_t a, uint64_t b) const { return a >= b ? a - b : a + (NK - b); }
    uint64_t mul(uint64_t a, uint64_t b) const;
};
// R: the wanted precision above the scale; K = R + E is checked against ell^K < 2^62.
// with_mu = false (P2x): only the e1 / ek tables; mu stays all zero (skips the O(units^2) convolution).
EisLadic BuildEisLadic(uint32_t n, uint64_t ell, uint32_t M, uint32_t R, unsigned threads, bool with_mu = true);
// ell B_s mod ell^K (s = 0 .. N) exactly (tangent numbers; v(B_s) >= -1).
std::vector<uint64_t> EllBernoulliModEllPower(uint32_t N, uint64_t ell, uint32_t K);
// The level-M polar element of pole branch `br` (scaled like L.mu): (1-ell)(1-ell^{k-1})(B_k/k) ell^{-M} N, N = sum_i u^i [gamma^i]
// (branch -1) or sum_i u^{-(k-1)i} [gamma^i] with an extra 1/(k-1) (branch k-1); zero vector off the pole branches.
// res_scale: multiply the residue by this unit (NEGATIVE CONTROLS: 1 + ell, -1); 1 in production.
std::vector<uint64_t> PolarElement(const EisLadic& L, uint32_t br, int64_t res_scale_num = 1);

// An ell-adic number as (v, unit mod ell^P); zero = exact zero at the working precision.
struct Lad {
    int32_t v = 0;
    uint64_t unit = 0;
    bool zero = false;
};
// Res = (1-ell)(log_ell(u)/ell)(1-ell^{k-1}) B_k/k mod ell^P (B_k/k exact).
Lad Residue(uint32_t k, uint64_t ell, uint32_t P);
// Res / (u^r - 1)
Lad PolarValue(const Lad& Res, uint64_t ell, uint64_t r, uint32_t P);
Lad LadMul(const Lad& a, const Lad& b, uint64_t ell, uint32_t P);
// x * ell^S mod ell^{S + W} (requires v + S >= 0 or x zero); for comparisons modulo ell^W.
uint64_t LadScaled(const Lad& x, uint64_t ell, uint32_t S, uint32_t W);

} // namespace eis
} // namespace lvalue
} // namespace hecke_ext

#endif // HECKE_EXT_LVALUE_EISENSTEIN_MEASURE_H
