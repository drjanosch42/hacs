// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

#pragma once
#ifndef HECKE_EXT_LVALUE_HIDA_PAIRING_H
#define HECKE_EXT_LVALUE_HIDA_PAIRING_H

// ===========================================================================
// (hecke-extensions slot 19, WP-BIL) HIDA'S GL2 x GL2 PAIRING ON THE 7-VERTEX MODEL -- the core of the `bilinear` kind of
// [E] def:lvroute (rmk:lvshapes row (H)), per docs/implementation/slot19/R19-9_report.md Sec. 6 (SPEC) and its Python
// prototype docs/implementation/slot19/r199/ (r199lib.py, pairlib.py, small.py, cell258.py), mirrored function by function.
//
// CONVENTIONS (R19-9 N1-N7).  V = Sym^n Z^2, u_t = x^{n-t} y^t, rho = rho_n ([C] def:rhon, column action, a monoid hom on
// Mat_2(Z)); D = Hom(V, Z) with the dual basis, <v, lam> = sum_t v_t lam_t, and rho^v(B) := rho(adj B)^T ([C] def:contragredient,
// = FibreRepresentation::ContragredientN).  Gamma = PSL_2(Z) = <S> * <U>, S = [[0,-1],[1,0]], U = [[0,-1],[1,1]]; n even, ell >= 5.
//   * H^1_c(V) = W := ker(1 + S) cap ker(1 + U + U^2) in V (R19-8 P1), P = Phi({0, oo});
//   * H^1(Gamma, D) = D / (D^S + D^U) (Mayer-Vietoris of the amalgam): a cocycle (c(S), c(U)) is normalised to c(S) = 0 by the
//     coboundary of -c(S)/2, then c(U) = (rho^v(U) - 1) y with y = -(2 + rho^v(U)) c(U) / 3 (N2);
//   * THE PAIRING (N3): <Phi, c> := <P, y_c>.  It identifies W with Hom(H^1(D)/tors, Z_ell) for every even n (perfect on free
//     parts, PROVED in R19-9 Sec. 1.2) and equals -1 x the cone-level cup form P_1 of Hecke/HeckePairing.h (CupEvaluationForm),
//     which the validation's arm (G1) re-checks on the C++ PoincareDualityPairing;
//   * T_p on W by Manin (R19-8), on H^1(D) by the group-cocycle formula (T_p c)(g) = sum_i rho^v(adj alpha_i) c(g_i),
//     alpha_i g = g_i alpha_j (isign), re-normalised -- the two are computed independently, and their adjunction
//     A_D(T_p) = A_c(T_p)^T is a CHECK, never an input;
//   * Hida's self-dual form for n + 1 < ell: G_!(P, Q) = <P, theta iota_!(Q)>, theta(u_s) = (-1)^s u_{n-s}^v / C(n, s),
//     iota_!(P) = -(1 + 2U) P / 6 (the y-picture of P -> (P, P)).
//
// ARITHMETIC.  Z/ell^M with q = ell^M < 2^62 (uint64_t residues, unsigned __int128 products); no bignum, no float.  Everything is
// host C++, single-threaded, no file, no env, no golden object.  ENGINE_CHECK for invariants (a failing one is a defect of THIS
// unit, reported as a validation failure).
// ===========================================================================

#include "../ZellK.h"

#include <cstdint>
#include <map>
#include <string>
#include <vector>

namespace hecke_ext {
namespace lvalue {
namespace hidapair {

using Vec = std::vector<uint64_t>;
using Mat = std::vector<Vec>;   // row-major

// ---- Z/ell^M and integral 2x2 matrices: the shared kit (S24 CL-G1) ---------------------------------------------------
// Zq(ell, M) checks ell^M <= 2^62 (FibreOps::eval accumulates eight u128 products); its precision is Zq::m.
using Zq = zk::Ring;
using M22 = zk::M22;
using zk::Adj;
using zk::Mul;

enum class Fibre : uint8_t { V = 0, D = 1 };

// rho_n, rho_n^T and rho^v on vectors, by L D U (R19-8 act / R19-9 rhoT), O(n^2) each.
class FibreOps {
public:
    FibreOps(const Zq& R, uint32_t n);
    uint32_t n() const { return n_; }
    const Zq& R() const { return R_; }
    Vec act(const M22& A, const Vec& v) const;      // rho(A) v
    Vec actT(const M22& A, const Vec& lam) const;   // rho(A)^T lam
    Vec dact(const M22& A, const Vec& lam) const { return actT(Adj(A), lam); }
    Vec on(Fibre f, const M22& A, const Vec& v) const { return f == Fibre::V ? act(A, v) : dact(A, v); }
    uint64_t eval(const Vec& v, const Vec& lam) const;
    uint64_t binom(uint32_t t, uint32_t s) const { return binom_[t][s]; }   // C(t, s) mod q
    // The full (n+1) x (n+1) matrix of rho(A) (rows), for the convention gate against the pipeline's builders.
    Mat matrix(Fibre f, const M22& A) const;

    Vec add(const Vec& u, const Vec& v) const;
    Vec sub(const Vec& u, const Vec& v) const;
    Vec scale(uint64_t s, const Vec& u) const;

private:
    Zq R_;
    uint32_t n_ = 0;
    std::vector<Vec> binom_;   // binom_[t][s] = C(t, s) mod q
    Vec upper(uint64_t b, const Vec& c) const;
    Vec lower(uint64_t g, const Vec& c) const;
    Vec diag(uint64_t a, uint64_t d, const Vec& c) const;
    Vec upperT(uint64_t b, const Vec& lam) const;
    Vec lowerT(uint64_t g, const Vec& lam) const;
    Vec sflip(const Vec& c) const;   // rho(S) c, = rho(S)^T on the dual side
    Vec sflipT(const Vec& c) const;
};

// ---- linear algebra over Z/ell^M ------------------------------------------------------------------------------------
struct SmithResult {
    std::vector<uint32_t> pivot_valuations;   // one per pivot found (valuation < M)
    std::vector<Vec> kernel;                  // saturated kernel basis (columns of the unimodular transform at the zero pivots)
};
SmithResult SmithKernel(const Zq& R, const Mat& rows, uint32_t ncols);
// Coordinates of each vector in a saturated basis, and the residual valuation (M = exact membership).
std::vector<std::pair<Vec, uint32_t>> CoordsInBasis(const Zq& R, const std::vector<Vec>& basis, const std::vector<Vec>& vecs);
using zk::MatMul;                           // Mat MatMul(const Zq&, const Mat&, const Mat&)
Mat Transpose(const Mat& A);
// low -> high, monic (Berkowitz, division-free; the shared kit).
inline Vec Charpoly(const Zq& R, const Mat& A) {
    ENGINE_CHECK_MSG(!A.empty(), "hidapair::Charpoly: empty matrix");
    return zk::Charpoly(R, A.size(), [&](size_t i, size_t j) { return A[i][j]; });
}
inline uint32_t RankModEll(const Zq& R, const Mat& A) {   // rank over F_ell
    return zk::FieldRank(A, A.empty() ? 0u : static_cast<uint32_t>(A[0].size()), R.ell);
}
Mat Block(const Mat& A, uint32_t lo, uint32_t hi);
Mat ReduceMat(const Zq& R, const Mat& A);    // entries mod q
// The matrix of t on the span of the saturated basis B (columns = coordinates of t b_j), t-stable to precision
// ell^{M-2} (checked).
Mat Restrict(const Zq& R, const Mat& t, const std::vector<Vec>& B);

// ---- O = Z/ell^M[s]/(s^2 - c1 s - c0) (or Z/ell^M itself) -----------------------------------------------------------
struct OElt {
    uint64_t a = 0, b = 0;
};
class QuadO {
public:
    // trivial: O = Z/ell^M (e = 1, v_pi = v_ell); else s^2 = c1 s + c0, Eisenstein (v(c0) = 1, ell | c1): e = 2, s a uniformiser.
    static QuadO Trivial(const Zq& R);
    static QuadO Eisenstein(const Zq& R, uint64_t c0, uint64_t c1);
    const Zq& R() const { return R_; }
    bool trivial() const { return trivial_; }
    uint32_t e() const { return trivial_ ? 1u : 2u; }
    OElt mul(const OElt& x, const OElt& y) const;
    OElt add(const OElt& x, const OElt& y) const { return {R_.add(x.a, y.a), R_.add(x.b, y.b)}; }
    OElt sub(const OElt& x, const OElt& y) const { return {R_.sub(x.a, y.a), R_.sub(x.b, y.b)}; }
    OElt scalar(uint64_t x) const { return {x % R_.q, 0}; }
    uint32_t vpi(const OElt& x, uint32_t cap) const;   // v_pi, capped at e * cap
    OElt div_pi(const OElt& x) const;                  // x / pi for x in pi O (one ell-digit of precision lost per call)
    uint64_t c0() const { return c0_; }
    uint64_t c1() const { return c1_; }

private:
    Zq R_;
    bool trivial_ = true;
    uint64_t c0_ = 0, c1_ = 0;
};

// The primitive generator of the lambda-eigenline of A (given the OTHER roots `others` of A's charpoly, or the cofactor Q_lambda
// when `cofactor` is non-empty): the column of prod(A - mu) of least content, divided by pi^content.  Returns (g, content,
// v_pi(t g - lambda g) as an exactness check).
struct EigenLine {
    std::vector<OElt> g;
    uint32_t content = 0;
    std::vector<uint32_t> column_contents;
    uint32_t eigen_check = 0;
};
EigenLine EigenLineOf(const QuadO& O, const Mat& A, const OElt& lambda, const std::vector<OElt>& others, uint32_t cap);
// Over O for a root rho of a monic quartic-or-any P with Q_rho(t) = P(t)/(t - rho) (cell258.eigvecs).
EigenLine EigenLineByCofactor(const QuadO& O, const Mat& A, const Vec& P, const OElt& rho, uint32_t cap);
OElt PairO(const QuadO& O, const std::vector<OElt>& g, const Mat& Gram, const std::vector<OElt>& h);

uint64_t SqrtModPrimePower(const Zq& R, uint64_t a);   // a a non-zero square unit mod ell; ENGINE_CHECKs

// ---- one (n, ell, M) cell ---------------------------------------------------------------------------------------------
class HidaPairingCell {
public:
    // primes: the Hecke primes whose A_c / A_D are built (2, 3, 5 by default).  Throws std::logic_error at odd n or ell < 5.
    HidaPairingCell(uint32_t n, uint64_t ell, uint32_t M, const std::vector<uint32_t>& primes = {2, 3, 5});

    uint32_t n() const { return n_; }
    const Zq& R() const { return R_; }
    const FibreOps& F() const { return F_; }
    // W = H^1_c(V): geo-even basis first (ne vectors), then geo-odd (no); saturated.
    const std::vector<Vec>& W() const { return Wb_; }
    uint32_t ne() const { return ne_; }
    uint32_t no() const { return no_; }
    const std::vector<uint32_t>& torsion_even() const { return tors_e_; }
    const std::vector<uint32_t>& torsion_odd() const { return tors_o_; }
    // y_j in D with <w_i, y_j> = delta_ij (a basis of H^1(D)/tors, dual to W).
    const std::vector<Vec>& Y() const { return ys_; }
    // Matrices in the column convention T x_j = sum_i A[i][j] x_i: A_c on W, A_D on H^1(D) in the dual coordinates.
    const Mat& Ac(uint32_t p) const;
    const Mat& AD(uint32_t p) const;
    const Mat& iotaD() const { return iotaD_; }
    bool has_prime(uint32_t p) const { return Ac_.count(p) != 0; }

    // The pairing <P, y> on representatives (V x D evaluation), and the dual coordinates (<w_i, y>)_i of a class of H^1(D).
    uint64_t pair(const Vec& P, const Vec& y) const { return F_.eval(P, y); }
    Vec dual_coords(const Vec& y) const;
    // N2 / N4 / N5 on either fibre.  normalise() refuses a non-cocycle (std::logic_error).
    Vec normalise(Fibre f, const Vec& xS, const Vec& xU) const;
    Vec hecke_y(Fibre f, const Vec& y, uint32_t p) const;
    Vec iota_y(Fibre f, const Vec& y) const;
    Vec hecke_symbol(Fibre f, const Vec& P, uint32_t p) const;   // T_p on modular symbols by Manin
    Vec theta(const Vec& P) const;                               // V -> D, n < ell
    Vec iota_shriek(Fibre f, const Vec& P) const;                // W -> H^1, the y-picture
    uint64_t r198_form(const Vec& P, const Vec& Q) const;        // R19-8's Haberland form B(P, Q), n < ell
    // Saturated basis of H^1_c on the chosen fibre and geometric sign (V: W^{par}; D: W_D^{par}); torsion valuations out.
    std::vector<Vec> sign_basis(Fibre f, uint32_t parity, std::vector<uint32_t>* torsion = nullptr) const;

    bool adjoint_ok(uint32_t p) const;     // A_D(T_p) == A_c(T_p)^T exactly
    bool iota_ok() const;                  // <iota P, iota y> = -<P, y>
    double seconds() const { return seconds_; }

private:
    uint32_t n_ = 0;
    Zq R_;
    FibreOps F_;
    std::vector<Vec> Wb_, ys_;
    uint32_t ne_ = 0, no_ = 0;
    std::vector<uint32_t> tors_e_, tors_o_;
    std::map<uint32_t, Mat> Ac_, AD_;
    Mat iotaD_;
    double seconds_ = 0;
};

// ---- readings ---------------------------------------------------------------------------------------------------------
// One eigen-system's valuations at a small cell (R19-9 Sec. 3 / small.py run()): v_pi of the mixed pairing on both sign pairs,
// Hida's H^1_! value (n + 1 < ell), and the congruence-module lengths read from A_c ALONE (the lemma's right-hand side, Q7).
struct EigenReading {
    std::string name;
    uint32_t e = 1;                    // ramification of O (v_pi units)
    uint32_t mixed_even = 0;           // v_pi <delta_c^{geo-even}, h^{geo-odd}>
    uint32_t mixed_odd = 0;            // v_pi <delta_c^{geo-odd}, h^{geo-even}>
    bool hida_defined = false;
    uint32_t hida = 0;                 // v_pi <delta_!^+, delta_!^->  = v(G_!(g^+, g^-)) - content of g^{even} mod p_0
    uint32_t cong_even = 0;            // v_pi(prod_{mu != lambda}(lambda - mu)) - content, on W^{geo-even}
    uint32_t cong_odd = 0;             // the same on W^{geo-odd} (= H^1_!^{odd})
    uint32_t eigen_check = 0;          // min v_pi(A g - lambda g) over the four eigenvectors
    bool t5_eigen = true;              // g^{even}, g^{odd} are T_5-eigen to the precision (Q6)
};
struct SmallCellReading {
    uint32_t n = 0;
    uint64_t ell = 0;
    uint32_t M = 0;
    uint32_t ne = 0, no = 0;
    bool w_torsion_free = true;
    bool adj2 = false, adj3 = false, adj5 = false, iota = false, t2t5_commute = false;
    bool hida_applicable = false;      // n + 1 < ell
    bool G_is_minus_sixth_B = false;   // G_! == -1/6 B_{R19-8}
    std::vector<uint32_t> G_pivots;    // Smith valuations of G_! on W (radical: rank W - #pivots)
    uint32_t G_radical = 0;
    std::string kind;                  // "Z" (dim S = 1), "split", "ramified"
    Vec charpoly_odd;                  // T_2 on W^{geo-odd} (= S_k), low -> high
    std::vector<EigenReading> systems;
    double seconds = 0;
};
SmallCellReading ReadSmallCell(uint32_t n, uint64_t ell, uint32_t M);

// (258, 37) mod 37^M: the m = (37, T_2 - 29) blocks (R19-9 Sec. 4) and P9.  P = R19-1's quartic, P_i its two Eisenstein factors
// mod 37^8 (frozen data of R19-1 / R19-8 cell258.py, cited, not recomputed here).
struct BlockEigen {
    std::string name;                  // f_(i, rho) / f_(i, rho')
    uint32_t mixed = 0;                // v_pi <g_f, h_f> on W_m x H^1(D)_m
    uint32_t p9 = 0;                   // v_pi <g_f^c, h_f^!> on W_m x H^1_!(D)_m (the opposite W sign, as cell258.p9)
    uint32_t vpi_dP = 0;               // v_pi P'(rho)
    uint32_t content_c = 0;            // min content of Q_rho(t) e_j on W_m
    uint32_t eigen_check = 0;
};
struct BlockSign {
    std::string name;                  // geo-even / geo-odd (the W sign)
    uint32_t rank_c = 0, rank_D = 0;
    bool charpoly_c_is_P = false, charpoly_D_is_P = false;
    uint32_t tr1 = 0, tr2 = 0, tr3 = 0;                // ranks of t, t^2, t^3 mod 37 on W_m
    uint32_t dr1 = 0, dr2 = 0, dr3 = 0;                // the same on H^1(D)_m
    std::vector<uint32_t> gram_pivots;                 // Smith valuations of the Gram W_m x H^1(D)_m
    bool gram_unimodular = false;
    std::vector<BlockEigen> forms;
    // P9 (indexed by the W sign whose dual block receives H^1_!(D))
    uint32_t p9_rank_WD = 0;
    bool p9_intertwines = false;
    bool p9_charpoly_is_P = false;
    std::vector<uint32_t> p9_index_pivots;            // Smith valuations of [H^1(D)_m : H^1_!(D)_m]
};
struct Cell258Reading {
    uint32_t M = 0;
    uint32_t ne = 0, no = 0;
    bool adj2 = false, adj3 = false, adj5 = false, iota = false;
    uint32_t HD_free_rank = 0;
    std::vector<uint32_t> HD_torsion;
    BlockSign sign[2];                 // [0] geo-even, [1] geo-odd
    double seconds = 0;
};
Cell258Reading ReadCell258(uint32_t M);
// R19-1's P (low -> high, mod 37^8) and its factors P_1, P_2 (low -> high, monic quadratics).
const std::vector<int64_t>& R191Quartic();
const std::vector<int64_t>& R191Factor(uint32_t i);

} // namespace hidapair
} // namespace lvalue
} // namespace hecke_ext

#endif // HECKE_EXT_LVALUE_HIDA_PAIRING_H
