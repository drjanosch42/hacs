// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

#pragma once
#ifndef INTEGRAL_DESCENT_HECKE_Z_CROSSED_HOM_ORACLE_H
#define INTEGRAL_DESCENT_HECKE_Z_CROSSED_HOM_ORACLE_H

// ===========================================================================
// Slot-26 card Z3b: an M oracle at ell = 2, 3 that does not use T~_Iso (docs/implementation/slot26/Z3b_design.md v0.2, audit
// Z3b_design_audit.md).  [C] = docs/latex/gl2_contract.tex, cited by label: eq:barcorrected, prop:barformula, prop:ltwothreescalar (I1),
// prop:descentN (2), (5), def:rhon, def:isomodeltree, prop:isomodeltree (2)(a), (4).
//
// THE MODEL.  H^1(SL_2(Z); M_N), M_N = Ind_{Gamma_0(N)}^Gamma Sym^n Z^2 with the block-monomial action (g.v)_j = rho_n(h_j(g)) v_{j.g},
// h_j(g) = w_j g w_{j.g}^{-1} (w_j = Gamma0LevelN::rep(j), w_1 = I), as crossed homomorphisms f given by (f(S), f(U)).  The free tree
// carrier Iso_T (C^1 = S-cell | U-cell | edge, each M_N) and the crossed homomorphisms are compared by the closed-form maps
//     kappa (a, b, z) = (a, b + (1 - U) z),        kappa' (f_S, f_U) = (f_S, f_U, 0)          (design Sec. 1.2)
// -- mutually inverse on H^1 over Z, no lift and no homotopy beyond (0, z).
//
// THE OPERATOR.  T^x f := (inverse Shapiro) o T^A o (restriction), A = Gamma_0(N), p !| N, with [C, eq:barcorrected] written for A:
//     (T^A phi)(gamma) = sum_{j=0}^{p} rho_n(delta_{gamma*j}) phi(delta_{gamma*j}^{-1} gamma delta_j),
//     delta_j = [[p, -j], [0, 1]] (j < p), delta_p = [[1, 0], [0, p]]   (the adjugates of the usual right-coset representatives);
// restriction phi(t) = f(t)_{j0} (w_{j0} = I), inverse Shapiro f'(x)_j = (T^A phi)(h_j(x)), x in {S, U}.  phi(t) is evaluated through
// a word of t in the letters S, U, T, T^{-1} (Euclid; the word is multiplied back and ENGINE_CHECKed) by a Horner scheme on the
// (n+1) x (n+1) blocks along the coset path j0, j0.x_1, ...: only block j0 is ever needed (design v0.2 Sec. 3, N3).
//
// NOT USED (the point of the card): IsoHeckeN / T~_Iso and its term list, TreeTransport::apply_hecke, Phi^Z_T / Psi^Z_T, HeckeCosetTables,
// RhoNZ, rho_n_cubic.  The oracle's rho_n is its own (the binomial formula of [C, def:rhon]); the validation compares it with RhoNZ.
// Pure host C++, residues modulo Q < 2^32 (u64 storage, __int128 products), no state, no env, serial.  Validation-only: nothing in
// production calls it.
// ===========================================================================

#include <cstdint>
#include <map>
#include <array>
#include <vector>

#include "../../Hecke/ExactInt.h"
#include "../../NumberField/BigInt.h"

namespace integral_descent {
namespace xhom {

using hecke::Mat22;

// ---- the oracle's own rho_n ([C, def:rhon]): entry (t, j) = [y^t] (a + c y)^{n-j} (b + d y)^j, row-major (n+1) x (n+1) ----------
std::vector<uint64_t> SymRhoMod(const Mat22& g, uint32_t n, uint64_t Q);
std::vector<nf::BigInt> SymRhoExact(const Mat22& g, uint32_t n);

// ---- words: letters 0 = S, 1 = U, 2 = T, 3 = T^{-1}; g = x_1 x_2 ... x_L (ENGINE_CHECKed by multiplying back) --------------------
enum Letter : uint8_t { kS = 0, kU = 1, kT = 2, kTi = 3 };
Mat22 LetterMat(uint8_t x);
std::vector<uint8_t> Word(const Mat22& g);
Mat22 Adj(const Mat22& g);   // the adjugate [[d, -b], [-c, a]]

// the representatives delta_0 .. delta_p of A diag(1, p) A / A (p !| N)
std::vector<Mat22> StandardReps(uint32_t p);

// ---- the model mod Q -----------------------------------------------------------------------------------------------------------
// Vectors of M_N with k columns are row-major D x k (row = coset j * (n+1) + s); carrier 1-cochains are 3D x k (S-cell | U-cell | edge).
class CrossedModel {
public:
    CrossedModel(uint32_t N, uint32_t n, uint64_t Q);
    uint32_t N, n, m, d, D;
    uint64_t Q;
    // per letter x, per coset j (0-based): col[x][j] = j.x (0-based), h[x][j] = h_j(x) in Gamma_0(N), blk[x][j] = rho_n(h_j(x)) mod Q
    std::array<std::vector<uint32_t>, 4> col;
    std::array<std::vector<Mat22>, 4> h;
    std::array<std::vector<std::vector<uint64_t>>, 4> blk;

    // out = x . v (the full action of a letter on M_N)
    void act(uint8_t x, const std::vector<uint64_t>& v, uint32_t k, std::vector<uint64_t>& out) const;
    // kappa / kappa' (design Sec. 1.2)
    void kappa(const std::vector<uint64_t>& c, uint32_t k, std::vector<uint64_t>& fS, std::vector<uint64_t>& fU) const;
    std::vector<uint64_t> kappa_prime(const std::vector<uint64_t>& fS, const std::vector<uint64_t>& fU, uint32_t k) const;
};

struct HeckeStats {
    uint64_t words = 0, letters = 0, evals = 0, memo_hits = 0;
};

// y = kappa'(T^x f) for the crossed homomorphism f = (fS, fU) (D x k each), at p (p !| N), with the representative system `reps`
// (p + 1 matrices delta_j; each class A delta A must be hit exactly once, ENGINE_CHECKed).  iota = true uses rho_n(adj delta) in place of
// rho_n(delta) (a tooth: the wrong normalisation).  The result is 3D x k (edge component 0).
std::vector<uint64_t> HeckeCross(const CrossedModel& X, const std::vector<uint64_t>& fS, const std::vector<uint64_t>& fU, uint32_t k, uint32_t p,
                                 const std::vector<Mat22>& reps, bool iota = false, HeckeStats* st = nullptr);

}  // namespace xhom
}  // namespace integral_descent

#endif
