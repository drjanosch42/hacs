// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

#pragma once
#ifndef HECKE_EXT_HECKE_HIDA_LAMBDA_H
#define HECKE_EXT_HECKE_HIDA_LAMBDA_H

// ===========================================================================
// (hecke-extensions slot 19, WP-HL) THE TRUNCATED LAMBDA-ADIC ORDINARY HECKE
// ALGEBRA  T^ord(a) (x) Lambda/(ell^K, X^m).  Binding text: [U]
// docs/latex/hida_hecke_algebra.tex 0.3.0-pre.3 part 3 (sec:u-lambda):
// def:u-trunc, prop:u-truncbasic, lem:u-wtprod, thm:u-interp (sharp),
// prop:u-lambdaprod, lem:u-twistvalues, lem:u-lambdaone, prop:u-eistrunc,
// pr:u-lambda (U11)-(U13).  Spec: docs/implementation/slot19/R19-2_report.md
// Sec. 6; the research prototype it ports is docs/implementation/slot19/r192/
// (forms.py, algebra.py, lambda_trunc.py), whose JSON (r192/data/) is the
// oracle of HeckeLambdaValidation.
//
// OWNER RULING (2026-10-03): the PRIMARY input is the q-expansion arm (Q):
// ordinary forms-side Hecke matrices on e M_k(Z/ell^W) and e S_k(Z/ell^W)
// from q-expansions on the Miller basis.  The cohomological tower route
// (HeckeHidaTower, ~470 GiB / 25 h per weight at 691) is only a cross-check
// at 37 and 103 (HeckeLambdaValidation's arm (T)).
//
// Pipeline side only: pure host C++, STL + NumberField (BigInt / BigRational,
// exact Bernoulli numbers for the cross-check) + EngineCheck.h + Fnv.h.
// Writes nothing, no golden object, no digest of record.  Every refusal is an
// ENGINE_CHECK (std::logic_error), so a caller sees "REFUSED" as an exception.
//
// ARITHMETIC.  Residues mod ell^W live in unsigned __int128 with ell^W < 2^84
// (691^8 ~ 2^75.4, 103^10 ~ 2^66.9): a product is split into 42-bit halves
// (ZMod::mul).  The long q-expansion products are exact multimodular NTTs over
// three (or four) 62-bit primes p = c 2^25 + 1, chosen at run time by a
// deterministic Miller-Rabin, recombined by Garner directly mod ell^W; the
// prime count is chosen so that prod p > min(len) (ell^W - 1)^2 (ENGINE_CHECK).
// ===========================================================================

#include <cstdint>
#include <functional>
#include <map>
#include <string>
#include <utility>
#include <vector>

#include "NumberField/BigInt.h"

namespace hecke_ext {
namespace lambda {

using u128 = unsigned __int128;
using Vec = std::vector<u128>;
using Mat = std::vector<Vec>;   // row-major, square or n x s

std::string U128Text(u128 x);   // decimal

// Z / M, M = ell^W < 2^84 (ENGINE_CHECK).
struct ZMod {
    uint64_t ell = 0;
    uint32_t W = 0;
    u128 M = 0;
    std::vector<u128> pw;   // ell^0 .. ell^W
    ZMod() = default;
    ZMod(uint64_t ell, uint32_t W);
    u128 red(u128 x) const { return x % M; }
    u128 add(u128 a, u128 b) const { u128 s = a + b; return s >= M ? s - M : s; }
    u128 sub(u128 a, u128 b) const { return a >= b ? a - b : a + (M - b); }
    u128 neg(u128 a) const { return a ? M - a : 0; }
    u128 mul(u128 a, u128 b) const;           // a, b < M
    u128 pow(u128 a, uint64_t e) const;
    u128 inv(u128 a) const;                   // a a unit mod ell (ENGINE_CHECK)
    u128 from_i64(int64_t v) const;
    uint32_t val(u128 x, uint32_t cap) const; // v_ell of x in [0, M); x == 0 -> cap
};

// ---- dense linear algebra mod M (the prototype's helpers, ported) -------------
Mat Ident(size_t n);
Mat MatMul(const Mat& A, const Mat& B, const ZMod& R);
Vec MatVec(const Mat& A, const Vec& v, const ZMod& R);
// Berkowitz, division-free: det(Y - A) mod M, coefficients low -> high, monic.
Vec Charpoly(const Mat& A, const ZMod& R);
// Inverse of a square matrix invertible mod ell (unit pivots), mod M.
Mat InvUnitMatrix(const Mat& A, const ZMod& R);
// Greedy columns of P independent mod ell (the prototype's indep_cols_mod_l).
std::vector<size_t> IndepColsModEll(const Mat& P, size_t ncols, uint64_t ell);

// ---- (slot-26 Z6) block helpers, exported from HeckeHidaLambda.cpp (bodies unchanged; [U, sec:u-H2] code map) ----
// A - c I mod M.
Mat MAddScaledIdent(const Mat& A, u128 c, const ZMod& R);
// A basis Bm (n x s, independent mod ell): s rows of Bm independent mod ell and the inverse of that s x s minor (unit pivots).
std::pair<std::vector<size_t>, Mat> BasisData(const Mat& Bm, const ZMod& R);
// The matrix of A on the A-stable summand spanned by the columns of Bm (ENGINE_CHECK: stable), mod M.
Mat Restrict(const Mat& A, const Mat& Bm, const std::vector<size_t>& rows, const Mat& Rinv, const ZMod& R);
// The Fitting split of t (N >= W n): {ker t^N, im t^N} as column bases (n x r) mod M; the kernel is the generalised
// kernel of t mod ell lifted to the Z/ell^W-summand.
std::pair<Mat, Mat> FittingSplit(const Mat& t, const ZMod& R);
// A nilpotent mod ell.
bool NilpotentModEll(const Mat& A, uint64_t ell);
// 1, t, ..., t^{r-1} independent mod ell (t generates an r-dimensional algebra mod ell).
bool GeneratesModEll(const Mat& t, uint64_t ell);
// The coordinates c with target = sum c_i t^i mod M (ENGINE_CHECK: t generates mod ell and target in Z[t]).
Vec PolyCoords(const Mat& target, const Mat& t, const ZMod& R);
// The exact Newton interpolant of (xs, vals) in Q, reduced mod M (ENGINE_CHECK: ell-integral), coefficients low -> high.
Vec NewtonMod(const std::vector<nf::BigInt>& xs, const Vec& vals, const ZMod& R);
// p(x) mod M (Horner).
u128 PEval(const Vec& p, u128 x, const ZMod& R);
// x_k = (1 + ell)^k - 1 mod ell^e (k >= 0).
nf::BigInt XOf(uint64_t ell, uint64_t k, uint32_t e);

// ---- the q-expansion input arm (Q) ---------------------------------------------
// One space (M_k or S_k) of the forms side: s = the ordinary rank, N = the power of
// T_ell used for the projector, matrices of T_p (p in primes) and of U (= T_ell
// on the ordinary part, valid as k - 1 >= W) in the basis of the columns of
// T_ell^N independent mod ell (Nakayama).
struct FormsSpace {
    uint32_t s = 0, N = 0;
    std::vector<std::string> names;          // "T2", "T3", ..., "U" (primes order, U last)
    std::map<std::string, Mat> mats;
    bool comm_ok = false;                    // T2 commutes with every operator
};
struct FormsWeight {
    uint64_t ell = 0;
    uint32_t k = 0, W = 0, dim_M = 0, dim_S = 0;
    uint64_t L = 0;                          // series length
    FormsSpace M, S;
    bool stable_ok = false;                  // T_p(M_k) beyond the triangular range, e M_k T_p-stable
    std::map<std::string, u128> eis;         // T_p: 1 + p^{k-1};  U: 1
    double wall_s = 0;
};
// ell >= 5 prime, k >= 4 even, W >= 1, ell^W < 2^84, k - 1 >= W (ENGINE_CHECK:
// U = T_ell mod ell^W on the ordinary part needs it), primes = the T_p read
// (p != ell, ascending).
FormsWeight ComputeFormsWeight(uint64_t ell, uint32_t k, uint32_t W, const std::vector<uint64_t>& primes);
// Delta = q prod (1 - q^n)^24 and E4^3 - E6^2 = 1728 Delta on n coefficients
// mod ell^W (schoolbook, independent of the NTT path).
bool SeriesSelfCheck(uint64_t ell, uint32_t W, uint32_t n);
// The NTT product against the schoolbook product (random inputs, several sizes).
bool NttSelfCheck(uint64_t ell, uint32_t W);
// The exact Berkowitz charpoly against exact BigInt determinants at 7 points.
bool CharpolySelfCheck();

// ---- (slot-25 H1) the ordinary Eisenstein block of one weight ---------------------
// The m_Eis-part of e M_k and e S_k cut out by the FULL residual system (RunClass's
// certified Fitting split of sum c_p (T_p - eis_p), p in primes[:4], every T_p in
// primes and U nilpotent mod ell after subtracting eis_p, resp. 1 -- ENGINE_CHECK):
// rank_M = 1 + g, rank_S = g, and every operator of the forms data restricted to it
// (names as FormsSpace::names), mod ell^W.  [U, sec:u-eislambda], [U, rmk:u-H0census].
struct EisBlock {
    uint32_t rank_M = 0, rank_S = 0;
    std::map<std::string, Mat> M, S;
};
EisBlock EisensteinBlock(const FormsWeight& w, const std::vector<uint64_t>& primes);

// ---- the Bernoulli side (lem:u-twistvalues) ------------------------------------
// B_k / k mod ell^prec for every even k in ks (k not = 0 mod ell - 1), by the
// Brent-Harvey tangent-number recurrence mod ell^{prec + V}, V = max v_ell(4^{k/2} - 1).
std::map<uint32_t, u128> BernoulliOverK(uint64_t ell, uint32_t prec, const std::vector<uint32_t>& ks);
// The same from WP-NF's exact Bernoulli numbers (nf::bernoulli_numbers), k <= kmax.
std::map<uint32_t, u128> BernoulliOverKExact(uint64_t ell, uint32_t prec, const std::vector<uint32_t>& ks);

// ---- thm:u-interp (with its precision bound ENFORCED) --------------------------
// The truncation mod (ell^K, (X - x_{centre_k})^m) of the function g with
// g(x_{k_j}) = values[j] mod ell^{Kp}, from N = values.size() weights.  REFUSED
// (ENGINE_CHECK) unless: the weights are consecutive in one class
// (k_j = k_0 + j(ell-1)), N <= ell, N >= K + m - 1 (lem:u-wtprod) and
// Kp >= K + m - 1 (thm:u-interp(2), sharp: R19-2 Sec. 4).  Returns m
// coefficients in Z = X - x_{centre_k}, low -> high, mod ell^K.
Vec InterpolateTruncation(uint64_t ell, const std::vector<uint32_t>& weights, const Vec& values, uint32_t Kp,
                          uint32_t K, uint32_t m, uint32_t centre_k);

// ---- the Lambda stage ----------------------------------------------------------
struct GeneralRoute {
    uint32_t A_log = 0, quot_log = 0, mingens = 0, coker_log = 0, J_log_index = 0;
    bool free = false, FY_zero = false, Y_generates = false, J_eq_G = false;
    std::vector<Vec> J;                       // Howell rows of the Hecke-side Eisenstein ideal
};
struct Component {                            // one factor's presentation Lambda_{K,m}[Y]/(F)
    bool present = false;
    std::string Y;                            // "T2 - eis_2(X)" | "T2"
    std::vector<Vec> F;                       // F[i] = coefficient of Y^i (polys in Z, mod ell^K)
    std::vector<std::string> op_names;
    std::map<std::string, std::vector<Vec>> ops;   // P_q(Y): coefficient polys of Y^0..Y^{r-1}
    // Eisenstein factor only:
    Vec G;                                    // G~ truncated
    bool F0_zero = false, F1_eq_G = false, G_eq_alpha = false;
    uint32_t cong_log = 0, cong_pred = 0;
    bool has_general = false;
    GeneralRoute general;
    // complement only: (U11) at the further weights (k, held, precision exponent)
    struct Further { uint32_t k; bool held; uint32_t prec; };
    std::vector<Further> further;
};
struct Truncation {
    uint32_t K = 0, m = 0, centre_k = 0, N = 0, W_mono = 0, W_gen = 0;
    Component E, c;
};
struct ClassResult {
    uint64_t ell = 0;
    uint32_t a = 0, W = 0, d = 0;
    std::vector<uint32_t> weights;
    uint32_t rank_E[2] = {0, 0}, rank_c[2] = {0, 0};
    std::string Y_E, Y_c;                     // "" = None
    std::vector<uint32_t> residue_degrees;    // of the complement (mod ell), sorted
    bool has_G0 = false;
    u128 G0_mod_l5 = 0;
    bool has_rung2 = false, has_rung3 = false;
    uint64_t rung2 = 0, rung3 = 0;
    u128 alpha = 0;
    uint32_t alpha_prec = 0;
    bool lambda_one = false;                  // v(G(x_k0) - G(x_k1)) == 1
    Vec G_at_weights;                         // G(x_k) mod ell^W at every class weight
    Vec F1_at_weights;                        // the Y^1-coefficient of charpoly(Y | V_E^M) mod ell^W, every class weight
    std::vector<Truncation> truncations;
};
struct ClassInput {
    uint64_t ell = 0;
    uint32_t W = 0;
    std::vector<uint64_t> primes;            // the T_p of the forms data (p != ell)
    std::vector<FormsWeight> weights;        // any order; the class's weights are selected
};
using Report = std::function<void(const std::string&)>;
// The prototype's run_class, ported: component split on the ordinary part
// (Fitting split of a generic sum of T_p - eis_p, the split certified: every
// T_p - eis_p and U - 1 nilpotent mod ell on V_E), monogenic route on both
// factors (precision K + m - 1 through InterpolateTruncation), general route
// on the Eisenstein factor at K + N - 1 (REFUSED if W is smaller), Bernoulli
// side, alpha, rungs, the KL value G~(0) mod ell^5.
ClassResult RunClass(const ClassInput& in, uint32_t a, uint32_t K, const std::vector<uint32_t>& ms,
                     const std::vector<uint32_t>& extra_centres, const Report& report);

// ---- the record hecke_lambda/v1 ------------------------------------------------
// One line per (ell, a, K, m, centre_k, component): "HECKE_LAMBDA ell= a= K= m=
// centre_k= N= W= route=Q component=E|c rank= gen= residue_degrees=[..] F=[[..]..]
// ops={T3:[[..]..],..,U:[[..]..]} cong_log= J=[[..]] alpha_mod=ell^e:<int>
// checks=<list> digest=0x<FNV-1a-64 of the line before ' digest='>".
// kl_ref: the expected G~(0) mod ell^5 (R19-4), or nullptr; "KL" is listed only when it agrees.
std::vector<std::string> RecordLines(const ClassResult& r, const std::string& route, const u128* kl_ref);

} // namespace lambda
} // namespace hecke_ext

#endif // HECKE_EXT_HECKE_HIDA_LAMBDA_H
