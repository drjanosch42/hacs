// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

#pragma once
#ifndef HECKE_EXT_LVALUE_WILD_CHARACTER_H
#define HECKE_EXT_LVALUE_WILD_CHARACTER_H

// ===========================================================================
// (hecke-extensions slot 21, P4) WILD CHARACTERS ON THE L-SIDE.  Spec docs/implementation/slot21/P4_spec.md (v0.1, predictions
// committed before this code ran); Python oracle docs/implementation/slot21/p4/ (p4lib.py, cal_*.py, out_cal.txt).
//
//   O = Z_ell[zeta], zeta of order ell^{m-1} (m >= 2), as (Z/ell^P)[z]/Phi_{ell^{m-1}}(z) on the power basis 1, z, .., z^{e-1}, e = phi(ell^{m-1})
//       (nf::CycloModRing's representation; OArith is its uint64 twin for ell^P < 2^62, and is checked against it; OArith128 (P4x,
//       P4_spec Sec. 10) the u128 one through engine::ModCtx<u128>, for W6 where ell^P >= 2^62, checked against OArith mod ell^{P64});
//   chi_{i,t}(a) = omega^i(a) zeta^{t L(a)},  <a> = a / omega(a) = u^{L(a)},  u = 1 + ell: conductor ell^m exactly when ell does not divide t;
//   F B_{r,chi} = sum_s C(r,s) (F^s B_s) S_{r-s},  S_q = sum_{a < F} chi(a) a^q,  F = ell^m  (the definition; exact B_s);
//   G_b(zeta^t u^r - 1) = sum_i g_i zeta^{t i} u^{i r}  (the level-M group ring of WP-MEAS at a wild point; = -B_{r, omega^{b-r} psi_t} / r).
// The complex side (P4_spec Sec. 1.5): the eigensymbols P^{+-} of the form spanning S_k (dim 1) exactly over Q, the functionals
// lambda_{chi,j} exactly per root of unity, and I(s) = (2 pi)^{-s} Gamma(s) L(f, chi, s) by the incomplete-Gamma series (nf::MpReal).
//
// STL + NumberField (BigInt, BigRational, MpReal, Cyclotomic) + the slot-19/20/21 L-value headers.  Pure host arithmetic; writes nothing;
// no golden object; no record.
// ===========================================================================

#include "BirchModularSymbol.h"

#include "NumberField/BigInt.h"
#include "NumberField/MpReal.h"

#include "../../../ModArithmetic/ModCtx.h"

#include <cstdint>
#include <map>
#include <vector>

namespace hecke_ext {
namespace lvalue {
namespace wild {

using u128 = unsigned __int128;

// (Z/ell^P)[z] / Phi_{ell^{m1}}(z), m1 >= 1, ell^P < 2^62.
struct OArith {
    uint64_t ell = 0, q = 0;
    uint32_t m1 = 0, P = 0;
    uint32_t order = 0;     // ell^{m1}: the order of zeta
    uint32_t s = 0;         // ell^{m1 - 1}
    uint32_t e = 0;         // (ell - 1) s: the degree (= the ramification index of pi = 1 - zeta)
    using Elt = std::vector<uint64_t>;

    Elt zero() const { return Elt(e, 0); }
    Elt one() const;
    Elt zeta_pow(uint64_t ex) const;                          // zeta^ex
    // dense[x] (x < order, entries already < q) -> the reduced element
    Elt reduce(std::vector<uint64_t> dense) const;
    Elt add(const Elt& a, const Elt& b) const;
    Elt sub(const Elt& a, const Elt& b) const;
    Elt scal(uint64_t c, const Elt& a) const;
    Elt mul(const Elt& a, const Elt& b) const;
    bool is_zero(const Elt& a) const;
    bool eq_mod(const Elt& a, const Elt& b, uint64_t modulus) const;  // a == b coordinatewise mod `modulus` (a divisor of q)
    uint32_t vpi(const Elt& a) const;                         // v_pi(a); e P for a == 0
    uint64_t mulq(uint64_t a, uint64_t b) const { return static_cast<uint64_t>(static_cast<u128>(a) * b % q); }
};
OArith MakeOArith(uint64_t ell, uint32_t m1, uint32_t P);

// L(a) = log_u <a> mod ell^{m-1} for a in [0, ell^m) (non-units: 0), and omega(a) mod ell^m.
std::vector<uint32_t> LogTable(uint64_t ell, uint32_t m);

// chi_{i,t} on [0, F): w[a] = omega^i(a) mod ell^P (0 at non-units), ex[a] = t L(a) mod ell^{m-1}.
struct WildChar {
    uint64_t ell = 0, F = 0;
    uint32_t m = 0;
    int64_t i = 0, t = 0;
    std::vector<uint64_t> w;
    std::vector<uint32_t> ex;
    OArith::Elt value(const OArith& O, uint64_t a) const;      // chi(a mod F) as an element
};
WildChar MakeWildChar(uint64_t ell, uint32_t m, int64_t i, int64_t t, uint32_t P, const std::vector<uint32_t>& logs);

// sum_a chi(a) x[a] for a table x indexed by a in [0, A) (A a multiple of F); x mod O.q.
OArith::Elt CharSum(const OArith& O, const WildChar& chi, const std::vector<uint64_t>& x);

// F B_{r,chi} for every r in [1, rmax] (index r), the definition; ellB[s] = ell B_s mod ell^P (EllBernoulliModEllPower).
std::vector<OArith::Elt> WildBernoulliFB(const OArith& O, const WildChar& chi, uint32_t rmax, const std::vector<uint64_t>& ellB);

// sum_i g_i zeta^{t i} u^{i r}  (g mod O.q; t may be negative).
OArith::Elt GroupRingAtWildPoint(const OArith& O, const std::vector<uint64_t>& g, int64_t t, uint64_t r);

// ---- (P4x) the 128-bit twin, for W6 where ell^P >= 2^62 (P4_spec Sec. 10) -------------------------------------
// (Z/ell^P)[z] / Phi_{ell^{m1}}(z) with u128 coefficients, products through engine::ModCtx<u128> (Montgomery-128, ell^P < 2^127).
struct OArith128 {
    uint64_t ell = 0;
    uint32_t m1 = 0, P = 0, order = 0, s = 0, e = 0;   // as OArith
    u128 q = 0;
    engine::ModCtx<u128> ctx;
    using Elt = std::vector<u128>;

    Elt zero() const { return Elt(e, 0); }
    Elt zeta_pow(uint64_t ex) const;
    Elt reduce(std::vector<u128> dense) const;               // dense[x], x < order, entries < q
    Elt add(const Elt& a, const Elt& b) const;
    Elt scal(u128 c, const Elt& a) const;
    Elt mul(const Elt& a, const Elt& b) const;
    u128 mulq(u128 a, u128 b) const { return ctx.mul(a % q, b % q); }
    u128 pow(u128 b, u128 ex) const;
};
OArith128 MakeOArith128(uint64_t ell, uint32_t m1, uint32_t P);

// omega^x(rho) mod ell^P, rho = 1 .. ell-1, every x mod ell-1: om[x][rho].
std::vector<std::vector<u128>> OmegaTable128(const OArith128& O);
// Res = (1-ell)(log_ell(u)/ell)(1-ell^{k-1}) B_k/k as v and a unit mod ell^P (exact B_k/k): eis::Residue's 128-bit twin.
struct Lad128 {
    int32_t v = 0;
    u128 unit = 0;
};
Lad128 Residue128(uint32_t k, const OArith128& O);

// F B_{r, omega^i psi_t} (F = ell^m, m = O.m1 + 1), the definition, for 1 <= r <= rmax.  The power sums S_q = sum_{a < F} chi(a) a^q are
// accumulated exactly (u256) once per Teichmuller class i, indexed by L(a); psi_{-t} = psi_t^{-1} is the relabelling zeta -> zeta^{-1}.
class WildBernoulliFB128 {
public:
    WildBernoulliFB128(const OArith128& O, uint32_t rmax, const std::vector<uint32_t>& logs);
    OArith128::Elt at(int64_t i, int64_t t, uint32_t r);
    const std::vector<u128>& ell_bernoulli() const { return ellB_; }

private:
    const OArith128& O_;
    uint64_t F_ = 0;
    uint32_t m_ = 0, rmax_ = 0;
    std::vector<uint32_t> logs_;
    std::vector<std::vector<u128>> om_;
    std::vector<u128> ellB_, FB_;                         // FB_[s] = F^s B_s mod ell^P
    std::vector<u128> apow_;                              // a^q mod ell^P, a-major: apow_[a (rmax+1) + q]
    std::map<uint32_t, std::vector<u128>> S_;             // i -> S[y (rmax+1) + q] = sum_{L(a) = y} omega^i(a) a^q
    const std::vector<u128>& sums(uint32_t i);
};

// sum_i g_i zeta^{t i} u^{i r}  (g mod O.q; t may be negative).
OArith128::Elt GroupRingAtWildPoint128(const OArith128& O, const std::vector<u128>& g, int64_t t, uint64_t r);

// ---- the complex side (W5) --------------------------------------------------------------------------------
// The normalised eigenform spanning S_k (k = 12: Delta; k = 16: Delta E_4) to q^{mmax}, exactly (int128; refuses past its bound).
std::vector<nf::BigInt> CuspQExp(uint32_t k, uint32_t mmax);
// The two eigensymbols P^+ (geometric-even coordinates), P^- (odd) of the form spanning S_k (dim S_k = 1), primitive integral, by
// BigRational elimination on (1 + S), (1 + U + U^2), T_2 - a_2; `kernel_dim` (2 expected).
struct ExactEigensymbols {
    std::vector<nf::BigInt> Pplus, Pminus;
    uint32_t kernel_dim = 0;
};
ExactEigensymbols ComputeExactEigensymbols(uint32_t k, const nf::BigInt& a2);
// rho_n([[M, -a], [0, 1]]) Phi({a/M, oo}) exactly (Manin path).
std::vector<nf::BigInt> ExactTwistVector(const std::vector<nf::BigInt>& P, uint32_t n, int64_t a, int64_t M);

} // namespace wild
} // namespace lvalue
} // namespace hecke_ext

#endif // HECKE_EXT_LVALUE_WILD_CHARACTER_H
