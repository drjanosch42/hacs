// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

#pragma once
// RibetField.h -- the degree-ell subfield F of the Ribet / Kummer field
// H = Q(zeta_ell, eta_i^{1/ell}) as an explicit polynomial over Z
// (hecke-extensions slot 21, T2; owner_q2_q4.md Q2 (b); the claims are stated
// for the referee in docs/implementation/slot21/T2_claim_for_F4.md). STL only.
//
// Setting. ell an odd prime, i even with ell | B_i, eta = eta_i =
// prod_{c=2}^{ell-1} u_c^{f_c} (u_c = (1 - zeta^c)/(1 - zeta), f_c = c^{-i} mod
// ell in (-ell/2, ell/2); Kummer.h / S3). eta is a real unit, a non-ell-th power
// with v_pi(eta^{ell-1} - 1) >= ell + 1, so H/Q(zeta) is cyclic of degree ell,
// unramified, and Gal(H/Q) = Z/ell x| (Z/ell)^x acts through omega^{1-i}; when
// that character is faithful (gcd(i - 1, ell - 1) = 1: 37 / 32, 691 / 12, 691 / 200)
// Gal(H/Q) is the Frobenius group of order ell(ell - 1) and F = H^C for a
// complement C has degree ell.
//
// The generator. For s in (Z/ell)^x let b_s be the REAL ell-th root of the real
// number sigma_s(eta) (at zeta -> z = e^{2 pi i/ell}); b_{-s} = b_s. The
// complement C_R containing complex conjugation gives
//     theta = Tr_{H/F}(beta),  conjugates  theta_j = sum_s b_s z^{j s^i}  (j in Z/ell),
// so theta_0 is real and theta_{-j} = conj(theta_j) (claim sheet C3). More generally
// psi(y) := Tr_{H/F}(y beta) for y in Q(zeta) has conjugates sum_s sigma_s(y) b_s z^{j s^i}.
//
// Two independent routes to P = charpoly(theta) in Z[X]:
//  * numerical (ribet_numeric + poly_from_conjugates): the b_s from the logarithms
//    log|2 sin(pi m/ell)| (MpReal.h), the theta_j, a product tree, rounding;
//  * exact modular (ribet_poly_modular): at primes q = 1 mod ell (q != 1 mod
//    ell^2) that split completely in H, the roots of P mod q are
//    sum_k r^{j e^k} beta_k with beta_0 an ell-th root of eta(r) and
//    beta_{k+1} = beta_k^e * gamma_R(r^{g^k}), gamma_R = zeta^m gamma the real
//    element with sigma_g(eta) = eta^e gamma_R^ell (g a primitive root mod ell,
//    e = g^i mod ell); CRT over enough primes for the bound (1 + sum_s |b_s|)^ell.
#include <cstdint>
#include <map>
#include <string>
#include <utility>
#include <vector>

#include "BigInt.h"
#include "Lll.h"
#include "MpReal.h"
#include "Polred.h"

namespace nf {

struct RibetFieldSpec {
    uint32_t ell = 0, i = 0;
    std::map<uint32_t, int64_t> f;  // f_c, c = 2 .. ell - 1 (kummer_exponents(ell, 1, i))
    int64_t F = 0;                  // sum_c f_c
    uint32_t g = 0;                 // the least primitive root mod ell
    uint32_t e = 0;                 // g^i mod ell
    std::vector<int64_t> gamma;     // c_d, d = 0 .. ell - 1 (entries 2 .. ell - 1 used): gamma = prod u_d^{c_d}
    uint32_t twist = 0;             // m: gamma_R = zeta^m gamma is real
    std::vector<int> sign;          // sign of sigma_s(eta), s = 1 .. ell - 1 (index s)
    std::vector<double> log_b;      // log b_s in double precision (index s)
    double bound_bits = 0;          // ell * log2(1 + sum_s |b_s|) * (1 + 1e-9) + 2: max |coeff P| < 2^bound_bits
    std::vector<uint32_t> spow;     // s^i mod ell (index s)
    int64_t n0 = 0;                 // eta mod pi as an integer in (-ell/2, ell/2)
};
// NF_CHECKs: ell prime >= 5, i even, gcd(i - 1, ell - 1) = 1, the gamma exponents
// integral (sigma_g(eta) / eta^e an ell-th power of cyclotomic units), gamma_R real
// in every embedding.
RibetFieldSpec ribet_field_spec(uint32_t ell, uint32_t i);

// Double-precision self-check of the complement relation b_s^e sigma_s(gamma_R) = b_{sg}
// for every s (log difference and sign); returns the largest |log difference|.
double ribet_complement_check(const RibetFieldSpec& S, bool* signs_ok);

struct RibetNumeric {
    unsigned prec = 0;
    std::vector<MpReal> b;          // b_s, s = 1 .. ell - 1 (index s; b[0] unused)
    std::vector<MpComplex> zpow;    // z^k, k = 0 .. ell - 1
    std::vector<MpComplex> theta;   // theta_j, j = 0 .. ell - 1
    double secs_logs = 0, secs_exps = 0, secs_theta = 0;
};
RibetNumeric ribet_numeric(const RibetFieldSpec& S, unsigned prec, unsigned threads);

// Conjugates (j = 0 .. ell - 1) of psi(y) = Tr_{H/F}(y beta), y = sum_a y[a] zeta^a.
std::vector<MpComplex> ribet_psi_conjugates(const RibetFieldSpec& S, const RibetNumeric& N, const std::vector<int64_t>& y);

// prod_j (X - x_j) for conjugates x_0 .. x_{ell-1} with x_0 real and x_{ell-j} = conj(x_j),
// rounded to Z[X] (constant term first); *max_log2_dist = max over the coefficients of
// log2 |c - round(c)| (the rounding margin; << -1 for a trustworthy result).
IntPoly poly_from_conjugates(const std::vector<MpComplex>& x, unsigned prec, unsigned threads, double* max_log2_dist);

struct RibetModular {
    IntPoly poly;
    std::size_t primes = 0;
    double modulus_bits = 0;
    uint64_t q_max = 0, q_min = 0;
    std::size_t candidates_tested = 0;  // primes q = 1 mod ell (not mod ell^2) examined
    std::size_t chain_failures = 0;     // beta_{ell-1} != beta_0 (tau^{ell-1} = 1 violated): must be 0
};
RibetModular ribet_poly_modular(const RibetFieldSpec& S, unsigned threads);

// Prediction (claim sheet C4) for a prime q != ell, q < 2^31: the number of roots of P mod q
// is 1 if q != 1 mod ell, else ell if eta(r) is an ell-th power mod q, else 0.
unsigned ribet_predicted_roots(const RibetFieldSpec& S, uint32_t q);
// deg gcd(X^q - X, P mod q) for a prime q < 2^31; -1 if P mod q is not squarefree.
int poly_root_count_mod(const IntPoly& P, uint32_t q);

// Generators of O_F (claim sheet C6). With n_0 = eta (mod pi) a rational integer, u = n_0 / beta
// satisfies u^ell = 1 (mod pi^ell), so x = (u - 1)/(1 - zeta) is integral and O_K[x] is
// lambda-maximal; Tr(O_K[beta]) is p-maximal for p != ell. H/F is tame, so
// O_F = Z[theta] + Tr_{H/F}(O_K[beta]) + Tr_{H/F}(O_K[x]); Tr(O_K beta^k) depends only on the
// class of k in (Z/ell)^x / <e>. Conjugate vectors (j = 0 .. ell - 1):
//   Tr(zeta^a beta^k)_j = sum_s z^{a s} (b_s z^{j s^i})^k,
//   Tr(zeta^a x^k)_j    = sum_s z^{a s} x_{s,j}^k,  x_{s,j} = (n_0 (b_s z^{j s^i})^{-1} - 1)/(1 - z^s).
// out[a][j], a = 0 .. ell - 2.
std::vector<std::vector<MpComplex>> ribet_trace_beta(const RibetFieldSpec& S, const RibetNumeric& N, uint32_t k,
                                                     unsigned threads);
// prec = 0: N.prec; otherwise the working precision of the sums (<= N.prec).
std::vector<std::vector<MpComplex>> ribet_trace_x(const RibetFieldSpec& S, const RibetNumeric& N, uint32_t k,
                                                  unsigned threads, unsigned prec = 0);

// D * (coordinates of y in the power basis of theta), rounded, for elements y given by their
// conjugates (Lagrange interpolation on the exact P); *max_log2_dist = rounding margin.
IntMat ribet_coords(const IntPoly& P, const std::vector<MpComplex>& theta, const std::vector<std::vector<MpComplex>>& elts,
                    const BigInt& D, unsigned prec, unsigned threads, double* max_log2_dist);

// floor(sqrt(n)), n >= 0.
BigInt bigint_isqrt(const BigInt& n);
// Upper-triangular (Hermite) basis rows of the lattice spanned by gens and D Z^n
// (all entries reduced mod D; Cohen GTM 138 Alg. 2.4.8 in incremental form).
IntMat hnf_mod(const IntMat& gens, const BigInt& D, std::size_t n);

// The lattice num_i(theta) / D (rows in the power basis of theta) with its exact trace form
// Tr(w_a w_b) (power sums of P) and discriminant.
struct RibetLattice {
    BigInt D;
    IntMat num;
    IntMat trace;
    BigInt disc;
};
RibetLattice ribet_lattice(const IntPoly& P, const BigInt& D, const IntMat& num, unsigned threads);
// round(2^shift T2(w_a, w_b)), T2(x, y) = Re sum_j x_j conj(y_j), from the numerical theta_j.
IntMat ribet_t2_gram(const RibetLattice& L, const std::vector<MpComplex>& theta, unsigned shift, unsigned prec,
                     unsigned threads);

// The maximal order O_F (claim sheet C6), built in well-conditioned coordinates:
//  (1) for each class kappa of (Z/ell)^x / <e>: LLL of [I | Minkowski(Tr(zeta^a beta^kappa))]
//      (a = 0 .. ell - 2) separates the exact relations (checked on the exact theta-coordinates)
//      from a basis of Tr(O_K beta^kappa); with 1 these give a basis of M_beta = Tr(O_K[beta]);
//  (2) [O_F : M_beta] = ell^v from disc(M_beta) (claim C2: v_ell disc F = ell - 2);
//  (3) the Tr(zeta^a x^k) (k = 1 .. ell - 1) in M_beta-coordinates times ell^v (numerical
//      inverse of the Minkowski matrix of M_beta, rounded), HNF modulo ell^v; the rows over
//      M_beta / ell^v, converted exactly to theta-coordinates over D (divisibility checked).
struct RibetMaxOrder {
    RibetLattice beta;                 // M_beta, rows over D in the theta power basis
    std::vector<uint32_t> kappas;      // class representatives used
    std::vector<std::size_t> ranks;    // rank of Tr(O_K beta^kappa) per class
    unsigned beta_shift = 0;           // the [I | 2^shift E] scale that separated the relations
    unsigned v = 0;                    // [O_F : M_beta] = ell^v
    double x_log2_dist = 0;            // rounding margin of the ell^v-scaled coordinates
    RibetLattice order;                // the result, rows over D in the theta power basis
    std::vector<std::pair<std::string, double>> secs;  // stage wall times
};
RibetMaxOrder ribet_maximal_order(const RibetFieldSpec& S, const RibetNumeric& N, const IntPoly& P, const BigInt& D,
                                  unsigned threads);

// POLRED on an order L (claim sheet C7): iterated LLL of rounded T2 (Minkowski) rows of L; alpha = the first reduced
// vector whose characteristic polynomial f (from its conjugates, rounded) is squarefree;
// d^2 = disc(f) / disc(L); the reduced basis as h_a(alpha) / d (Lagrange on the conjugates
// of alpha, rounded); exact checks; then T1's polred(f, h, d, reduced T2 Gram).
struct RibetPolred {
    IntMat H;                       // LLL transformation (rows: reduced basis in L's basis)
    unsigned passes = 0;            // LLL passes until H = I
    IntPoly f;                      // charpoly of alpha (numerical, rounded; checked exactly below)
    double f_log2_dist = 0;
    BigInt d;                       // [L : Z[alpha]]
    IntMat h;                       // d * (reduced basis) in the power basis of alpha
    double h_log2_dist = 0;
    bool trace_exact = false;       // the trace form of h / d (over f) equals H Tr_L H^T
    bool theta_exact = false;       // charpoly(theta expressed over alpha) == P: Q(theta) = Q(alpha)
    PolredResult pr;
    IntPoly best;
    BigInt best_disc;
    std::vector<std::pair<std::string, double>> secs;  // stage wall times
};
RibetPolred ribet_polred(const IntPoly& P, const std::vector<MpComplex>& theta, const RibetLattice& L, unsigned prec,
                         unsigned threads);

}  // namespace nf
