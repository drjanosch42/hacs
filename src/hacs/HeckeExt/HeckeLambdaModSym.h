// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

#pragma once
#ifndef HECKE_EXT_HECKE_LAMBDA_MODSYM_H
#define HECKE_EXT_HECKE_LAMBDA_MODSYM_H

// ===========================================================================
// (hecke-extensions slot 21, P8) ORDINARY COHOMOLOGY ALONG THE TOWER AS A
// LAMBDA-MODULE: truncated Lambda-adic modular symbols.  Binding text: [U]
// docs/latex/hida_hecke_algebra.tex v0.5.0-pre.2 part 4 (sec:u-msym):
// def:u-msymk, lem:u-msymsigns, prop:u-msymfree (with (BF) over every
// b mod ell - 1, F4 M1), lem:u-msymmass (the operator form of the MTT measure),
// prop:u-msymnorm (the normaliser of record), pr:u-msym (U15)-(U17).  Spec:
// docs/implementation/slot21/P8_spec.md v0.2 (referee F4,
// docs/implementation/slot21/P8_F4_referee_report.md).
//
// WHAT IT COMPUTES, per weight k (n = k - 2), per [C]-sign, per etale cusp
// component m = (ell, g(T_{p0})) of P7 (HeckeLambdaQExp):
//  * W^{par}(Z/ell^R) = ker(1 + rho(S)) cap ker(1 + rho(U) + rho(U)^2) on the
//    coordinates of geometric parity par ([C]-sign = - geometric parity), read
//    as the image of the reduction W(Z/ell^{R+E}) -> W(Z/ell^R) (torsion of
//    exponent <= E dies; F4 S2/S3), a free Z/ell^R-module (else E grows);
//  * the generalised g(T_{p0})-part of W, T_ell on that part ONLY (the cost
//    lever of the spec: never on all of W at ell >= 37), the ordinary part
//    O = im T_ell^N of it, and only then the freeness / rank reading (G1) --
//    the reordering of step 3 (F4 S2);
//  * on O: T_p (p <= B), T_ell, and U, the unit root of Y^2 - T_ell Y +
//    ell^{k-1} by Hensel (U = T_ell exactly when k - 1 >= R, G16);
//  * at m_Eis on the geometric-even sign: p_0 = u_0 - u_n spans a summand of O
//    (first basis vector), r = the H^1_c-primitive cusp eigensymbol, its depth
//    d (r == u p_0 mod ell^d exactly), w = (r - u p_0) / ell^d (prop:u-msymfree(3));
//  * the eigensymbol phi (cyclic over T_{p0} on S = O, or O / Z p_0), the
//    polynomials P_p with T_p = P_p(T_{p0}) on S, the operator-valued masses
//    mu_x(a + ell^m) = c_{U^{-m} x}(a, m) - ell^{k-2} c_{U^{-m-1} x}(a, m-1)
//    from WP-MEAS's c-tables (BuildMeasure, its scalar alpha unused), the
//    h-valued masses by the trace form of h_k = (Z/ell^R)[Y]/(F_k);
//  * the normaliser of record B_0, nu_B(x_k) = mu^h(B) / mu^h(B_0), and the
//    interpolation through WP-HL's InterpolateTruncation (precision enforced)
//    plus the full Newton interpolant's prediction at further weights mod
//    ell^{min(K', N)} (thm:u-interp(3));
//  * the gluing depth j_k of (U17)(a) on the dual of the D-fibre symbols
//    (H^1(Gamma; V)/tors = Hom(H^1_c(Gamma; D), Z_ell), WP-BIL's pairing) and,
//    at small n, directly on H^1(Gamma; V) = V / (V^S + V^U) (WP-BIL's amalgam
//    presentation on the V fibre).
//
// Pure host C++ on top of WP-LIN's symbols (BirchModularSymbol.h), WP-MEAS's
// c-tables (ModularSymbolMeasure.h), WP-BIL's fibre operations (HidaPairing.h),
// WP-LV3's joint block (MBlockReader.h), WP-HL / P7 (HeckeHidaLambda.h,
// HeckeLambdaQExp.h).  Writes nothing; no golden object; no digest of record.
// Every refusal is an ENGINE_CHECK (std::logic_error) or a named `refusal`.
// ===========================================================================

#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "HeckeHidaLambda.h"
#include "HeckeLambdaQExp.h"
#include "LValue/BirchModularSymbol.h"

namespace hecke_ext {
namespace lambda {
namespace msym {

namespace bm = hecke_ext::lvalue::birch;
using SVec = std::vector<uint64_t>;
using SMat = bm::Mat;   // column convention: T b_j = sum_i A(i, j) b_i

// ---- small dense helpers over Z/ell^R (exposed for the validation) -------------
SMat Identity(const bm::Ring& R, uint32_t n);
SMat MatPolyOf(const bm::Ring& R, const SVec& poly, const SMat& A);   // poly(A), poly low -> high
SMat MatPow(const bm::Ring& R, const SMat& A, uint64_t e);
bool InverseUnit(const bm::Ring& R, const SMat& A, SMat& out);       // false if not invertible
SMat Reduce(const SMat& A, uint64_t q);
SVec Reduce(const SVec& v, uint64_t q);
bool MatEq(const SMat& A, const SMat& B);
uint32_t RankModEll(const bm::Ring& R, const SMat& A);
bool NilpotentModEll(const bm::Ring& R, const SMat& A);
SMat AddScalarPublic(const bm::Ring& R, const SMat& A, uint64_t s);   // A + s I

// A basis of a free direct summand of (Z/ell^R)^len with exact coordinates.
struct FreeBasis {
    std::vector<SVec> b;          // basis vectors
    std::vector<uint32_t> rows;   // rows of a unit minor
    SMat minv;                    // inverse of that minor
    uint32_t rank() const { return static_cast<uint32_t>(b.size()); }
    bool coords(const bm::Ring& R, const SVec& w, SVec& c) const;   // false: w not in the span
};
// vecs a basis of a free direct summand (ENGINE_CHECK otherwise).
FreeBasis MakeFreeBasis(const bm::Ring& R, const std::vector<SVec>& vecs);

// ---- Hecke on symbols ------------------------------------------------------------
// T_p on a symbol of W (Manin, WP-LIN's formula).  An own implementation with a fast
// reduction for q < 2^32 (the cost of P8 is these calls); gated against bm::HeckeTp.
class SymbolHecke {
public:
    SymbolHecke(uint32_t n, const bm::Ring& R);
    SVec Tp(const SVec& P, uint64_t p) const;
    SVec apply(const bm::Mat2& A, const SVec& v) const;
    uint32_t n() const { return n_; }
    const bm::Ring& ring() const { return R_; }

private:
    uint32_t n_;
    bm::Ring R_;
    std::vector<uint64_t> binom_;
    uint64_t mul(uint64_t a, uint64_t b) const;
    SVec upper(uint64_t b, const SVec& v) const;
    SVec lower(uint64_t g, const SVec& v) const;
    SVec diag(uint64_t a, uint64_t d, const SVec& v) const;
    SVec sflip(const SVec& v) const;
};

// ---- the symbol space of one weight and one sign --------------------------------
struct SignSpace {
    uint64_t ell = 0;
    uint32_t n = 0, parity = 0, R = 0, E = 0;     // geometric parity; the reduction W(R+E) -> W(R)
    int c_sign = 0;                               // [C] sign = - (-1)^parity
    bm::Ring ring;
    std::shared_ptr<SymbolHecke> hecke;
    FreeBasis W;
    bool wrong_adjoint = false;                   // NEGATIVE CONTROL: HeckeTpWrongAdjoint for every T_p on W
    std::map<uint64_t, SMat> T;                   // T_p on the W basis (filled on demand)
    SVec ApplyT(const SVec& v, uint64_t p) const;
    const SMat& TonW(uint64_t p);
};
// R: the target precision; Emax: the largest extra precision tried (ell^{R+E} < 2^62).
SignSpace BuildSignSpace(uint64_t ell, uint32_t n, int c_sign, uint32_t R, uint32_t Emax, bool wrong_adjoint = false);

// The ordinary g-part of one sign space.
struct OrdBlock {
    uint32_t k = 0, n = 0, R = 0;
    int c_sign = 0;
    uint32_t w_rank = 0, g_rank = 0, ord_rank = 0;
    bool g_free = false, ord_free = false;
    bool has_p0 = false;                          // p_0 is the first basis vector
    bool p0_in_block = false;
    FreeBasis O;                                  // ambient vectors
    std::map<uint64_t, SMat> T;                   // T_{p0}, T_ell on O (others on demand)
    SMat Tell, U;
    bool U_eq_Tell = false, U_root_ok = false;
    uint32_t hensel_steps = 0;
    std::string refusal;
};
// The generalised g(T_{p0})-part (g monic mod ell, low -> high), T_ell on it only,
// O = its ordinary part, operators on O.  want_p0: put p_0 first if it lies in O.
OrdBlock OrdinaryPart(SignSpace& S, uint32_t k, uint64_t p0, const std::vector<uint64_t>& g, bool want_p0);
// The whole ordinary rank of W^{sign} (T_ell on all of W; small ell only).
uint32_t OrdinaryRankAll(SignSpace& S);

// ---- masses ----------------------------------------------------------------------
// c-tables of every O basis vector (WP-MEAS's BuildMeasure, its scalar alpha unused).
struct MassTables {
    uint32_t M = 0, k = 0;
    std::vector<std::vector<std::vector<uint64_t>>> c;   // c[j][m+1][a], m = -1 .. M
};
// mu_x(a + ell^m Z_ell), x in O coordinates, with the operator Uop (normally B.U) on O.
class OperatorMeasure {
public:
    OperatorMeasure(const bm::Ring& R, const MassTables& T, const SMat& Uop, bool ell_n_factor = true);
    uint64_t ball(const SVec& x, uint64_t a, uint32_t m) const;
    // the distribution relation sum_b mu(a + b ell^m, m+1) = mu(a, m) at every a < ell^m, m < M: failures
    uint32_t distribution_failures(const SVec& x) const;

private:
    bm::Ring R_;
    const MassTables* T_;
    std::vector<SMat> Uinv_;   // U^{-m}, m = 0 .. M + 1
    uint64_t lk2_ = 0;
};

// The unit balls of level 1 .. M in the order of the normaliser of record:
// M_0 = 1, 2, ...; b_0 = 1 .. ell^{M_0} - 1 with ell !| b_0.
struct Ball {
    uint64_t b;
    uint32_t M;
};

// ---- h_k = (Z/ell^R)[Y]/(F), F monic of degree r ----------------------------------
struct HRing {
    bm::Ring R;
    SVec F;                       // r + 1 coefficients, monic
    uint32_t r() const { return static_cast<uint32_t>(F.size() - 1); }
    SVec mul(const SVec& x, const SVec& y) const;
    bool inv(const SVec& x, SVec& out) const;      // false: not a unit
    bool unit(const SVec& x) const;                // x mod (ell, F): F == g mod ell irreducible -> x mod ell != 0
    SVec trace_gram() const;                       // Tr(Y^{i+j}), i, j < r, row-major
};
// mu^h with Tr(Y^i mu^h) = mus[i]; false if the Gram determinant is not a unit (refused).
bool HValuedMass(const HRing& h, const SVec& mus, SVec& out);

// ---- one (weight, sign, component) reading ------------------------------------------
struct ReadingRequest {
    uint64_t ell = 0;
    uint32_t k = 0, Kp = 0;                // output precision K'
    int c_sign = 0;
    uint64_t p0 = 0;
    std::vector<uint64_t> g;               // residual factor of T_{p0}
    bool eis = false;                      // the component through m_Eis (r = 1, residual root e_{p0})
    uint32_t D = 0;                        // extra precision for the depth (eis on the p_0 sign)
    uint32_t Emax = 2;
    uint32_t M = 2;                        // ball levels
    std::vector<uint64_t> primes;          // T_p, p <= B, p != ell
    bool wrong_adjoint = false;            // NEGATIVE CONTROL
    std::shared_ptr<SignSpace> space;      // optional: a cached sign space of the same (n, sign, R)
};
struct Reading {
    ReadingRequest req;
    uint32_t n = 0, R = 0, E = 0;
    uint32_t w_rank = 0, g_rank = 0, ord_rank = 0, r = 0;
    bool ord_free = false, has_p0 = false, p0_in_block = false;
    std::string refusal;                  // "" | boundary_component | not_cyclic | ...
    // the m_Eis p_0 sign
    uint32_t depth_d = 0, content = 0;
    SVec r_sym, w_sym;                     // O coordinates mod ell^{Kp}
    // phi and the operators on S, mod ell^{Kp}
    SVec phi;                              // O coordinates
    SVec F;                                // charpoly of T_{p0} on S
    std::map<uint64_t, SVec> P;            // T_p = P_p(T_{p0}) on S (r coefficients); key ell: U
    bool P_consistent = false;             // every T_p, U matrix on S equals P_p(T_{p0})
    bool U_eq_Tell = false, U_root_ok = false, lk2_vanishes = false;
    // (BF)
    bool bf_checked = false, bf_holds = false;
    int bf_fail_b = -1;
    // masses (O coordinates of phi and of y_i = T_{p0}^i phi), mod ell^{Kp}
    std::vector<Ball> balls;
    std::vector<SVec> muh;                 // per unit ball: the h-valued mass of phi (r coefficients)
    uint32_t dist_failures = 0;            // the operator masses of every y_i, every ball to M
    uint32_t p0_unit_mass_nonzero = 0;     // G7: unit balls with mu_{p0} != 0 (U = 1 on p_0)
    bool gram_unit = false;
    double seconds = 0;
    // retained for the gates (the measure of a vector in O coordinates)
    std::shared_ptr<SignSpace> space;
    std::shared_ptr<OrdBlock> block;
    std::shared_ptr<MassTables> tables;
    SMat TS_p0;                            // T_{p0} on S in the cyclic basis y_i
};
Reading ReadComponent(const ReadingRequest& req);
// The unit-ball masses of an arbitrary O vector (operator U of the block), reduced mod ell^{Kp}.
std::vector<uint64_t> UnitMasses(const Reading& rd, const SVec& x_O, const SMat* Uoverride = nullptr);

// ---- normalisation and interpolation ------------------------------------------------
// The normaliser of record: the first ball with mu^h(B_0) a unit at the centre reading; -1: no_unit_ball.
int FirstUnitBall(const Reading& centre);
// nu_B(x_k) = mu^h(B) / mu^h(B_0) in h_k (r coefficients) for every ball; false: B_0 not a unit at k.
bool Normalised(const Reading& rd, int b0, std::vector<SVec>& nu);
// The same prediction is linear in the values: lambda_a (mod ell^{Kp}) with prediction = sum_a lambda_a v_a.
std::vector<uint64_t> NewtonWeights(uint64_t ell, const std::vector<uint32_t>& weights, uint32_t Kp, uint32_t kprime);

// thm:u-interp's truncation as a linear map: trunc_t(v) = (sum_a lift(v_a) Lam_{t,a} mod ell^{K+m-1}) / ell^{m-1},
// t < m (Newton's divided differences recentred at x_c, the same arithmetic as WP-HL's InterpolateTruncation, which the
// validation compares on every level-1 ball).  Refuses (ENGINE_CHECK) as InterpolateTruncation does on the shape;
// Apply returns false where the result is not ell-integral.
struct TruncationMap {
    uint64_t ell = 0;
    uint32_t K = 0, m = 0, Kp = 0, centre_k = 0;
    std::vector<uint32_t> weights;
    std::vector<std::vector<u128>> lam;   // m x N, mod ell^{K + m - 1}
    bool Apply(const std::vector<uint64_t>& values, Vec& out) const;
};
TruncationMap MakeTruncationMap(uint64_t ell, const std::vector<uint32_t>& weights, uint32_t Kp, uint32_t K, uint32_t m, uint32_t centre_k);

// ---- (U17)(a): the gluing depth j_k on H^1 at m_Eis ---------------------------------------
struct GluingReading {
    uint32_t n = 0, M = 0;
    std::string route;               // "dual-D" | "amalgam-V"
    uint32_t block_rank = 0, ord_rank = 0;
    uint32_t j = 0;                  // v_ell [M : A_1 + A_2]
    uint32_t v_a_minus_e = 0;        // v(a_{p} - e_{p}) on the block (p = 2)
    uint32_t pivot = 0;              // Smith pivot of T_2 - e_2 on M (j = v(a - e) - pivot, the second reading)
    bool eis_in_plus = false;        // amalgam route: the Eisenstein class is [C]-plus (-iota_geo)
    bool shriek_in_cusp_line = false;// amalgam route: iota_!(the cusp symbol) lies on the cusp line
    std::string refusal;
    double seconds = 0;
};
GluingReading GluingDual(uint32_t n, uint64_t ell, uint32_t M);
GluingReading GluingAmalgam(uint32_t n, uint64_t ell, uint32_t M);
// The ordinary ranks of H^1(Gamma; V)^{[C]+} and ^{[C]-} on the amalgam presentation (T_ell on all of H^1; small n only).
std::pair<uint32_t, uint32_t> AmalgamOrdinaryRanks(uint32_t n, uint64_t ell, uint32_t M);

// ---- the record hecke_lambda_msym/v1 ----------------------------------------------------
struct RecordInput {
    uint64_t ell = 0;
    uint32_t a = 0, K = 0, m = 0, centre_k = 0, N = 0;
    uint32_t component = 0;
    int sign = 0;
    uint32_t rank = 0;
    uint64_t p0 = 0;
    int b0 = -1;
    uint32_t B0_M = 0;
    uint64_t B0_b = 0;
    bool unit_ok = false;
    std::vector<uint32_t> weights;
    std::vector<std::pair<Ball, std::vector<Vec>>> nu;    // per ball: the truncation of each Y-coefficient (m coeffs mod ell^K)
    std::vector<std::string> further;                      // "k':ok" | "k':fail@<v>"
    std::vector<int32_t> j;                                // gluing depths (-1 = not read)
};
// One line; level-1 balls printed, every ball in the digest (FNV-1a-64 of the line before ' digest=' plus the level-2 text).
std::string RecordLine(const RecordInput& in);

}  // namespace msym
}  // namespace lambda
}  // namespace hecke_ext

#endif  // HECKE_EXT_HECKE_LAMBDA_MODSYM_H
